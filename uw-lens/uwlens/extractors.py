"""Turn files into embeddable chunks.

Each supported file yields one or more `Chunk`s. Optional sidecar `<file>.meta.json`
supplies a title, source, caption/transcript (stand-in for OCR/ASR) and tags.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .textproc import chunk_text, detect_language, redact_phi

TEXT_EXT = {".txt", ".md"}
CODE_EXT = {".py", ".js", ".ts", ".sql", ".json", ".yaml", ".yml"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
SUPPORTED_EXT = TEXT_EXT | CODE_EXT | PDF_EXT | IMAGE_EXT | AUDIO_EXT | VIDEO_EXT

SOURCES = ("nih", "japan", "policy", "case", "other")
_SOURCE_HINTS = {
    "nih": ("nih", "medlineplus", "clinical", "reference"),
    "japan": ("japan", "mhlw", "nhi", "jp", "japanese"),
    "policy": ("policy", "policies", "manual", "guideline", "underwriting"),
    "case": ("case", "cases", "applicant", "aps", "claims", "evidence"),
}

AUDIO_SR = 16_000


@dataclass
class Chunk:
    id: str
    path: str
    title: str
    modality: str  # text | code | pdf | image | audio | video
    source: str
    language: str
    text: str  # redacted; BM25 + snippet + fallback embedding text
    locator: str = ""  # "p3", "chunk 2", "00:30-01:00"
    tags: list[str] = field(default_factory=list)
    media: Any = field(default=None, repr=False)  # embed payload, never persisted

    def record(self) -> dict:
        d = asdict(self)
        d.pop("media")
        return d


def modality_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return "text"
    if ext in CODE_EXT and not path.name.endswith(".meta.json"):
        return "code"
    if ext in PDF_EXT:
        return "pdf"
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    return None


def iter_files(folder: Path) -> Iterator[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts):
            if modality_of(p):
                yield p


def fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    meta = sidecar_path(path)
    if meta.exists():
        h.update(meta.read_bytes())
    return h.hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def load_sidecar(path: Path) -> dict:
    meta = sidecar_path(path)
    if not meta.exists():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def guess_source(path: Path, root: Path) -> str:
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]] if path.is_relative_to(root) else []
    for part in parts:
        for source, hints in _SOURCE_HINTS.items():
            if any(h == part or part.startswith(h) for h in hints):
                return source
    return "other"


def _chunk_id(path: Path, locator: str) -> str:
    return hashlib.sha1(f"{path.resolve()}::{locator}".encode()).hexdigest()[:20]


def extract(path: Path, root: Path, *, chunk_chars: int = 1200, redact: bool = True,
            audio_window_s: float = 30.0, load_audio: bool = False) -> list[Chunk]:
    """Extract chunks from one file. Raises on unreadable files; the indexer logs and skips."""
    modality = modality_of(path)
    if modality is None:
        return []
    meta = load_sidecar(path)
    title = str(meta.get("title") or path.stem.replace("_", " "))
    source = str(meta.get("source") or guess_source(path, root))
    if source not in SOURCES:
        source = "other"
    tags = [str(t) for t in meta.get("tags", [])]
    described = " ".join(str(meta.get(k, "")) for k in ("caption", "transcript")).strip()

    def clean(text: str) -> str:
        return redact_phi(text)[0] if redact else text

    def make(locator: str, text: str, media: Any = None, mod: str = modality) -> Chunk:
        text = clean(text)
        lang = str(meta.get("language") or detect_language(text or title))
        return Chunk(_chunk_id(path, locator), str(path), title, mod, source, lang, text, locator, tags, media)

    if modality in ("text", "code"):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if not meta.get("title"):
            title = (_heading(raw) or title) if modality == "text" else path.name
        return [make(f"chunk {i + 1}", c) for i, c in enumerate(chunk_text(clean(raw), chunk_chars))]

    if modality == "pdf":
        import pymupdf

        out = []
        with pymupdf.open(path) as doc:
            if not meta.get("title") and len(doc):
                title = _heading(doc[0].get_text()) or title
            for page_no, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                parts = chunk_text(clean(text), chunk_chars)
                for j, part in enumerate(parts):
                    loc = f"p{page_no}" if len(parts) == 1 else f"p{page_no}.{j + 1}"
                    out.append(make(loc, part))
        if not out and described:  # scanned PDF with sidecar OCR text
            out.append(make("ocr", described))
        return out

    label = f"{title}. {described}".strip()
    if modality == "image":
        return [make("", label, media=str(path))]
    if modality == "video":
        return [make("", label, media=str(path))]
    if modality == "audio":
        if not load_audio:
            return [make("", label, media=str(path))]
        return _audio_chunks(path, meta, label, audio_window_s, make)
    return []


def _audio_chunks(path, meta, label, window_s, make) -> list[Chunk]:
    """Split audio into fixed windows (16 kHz mono arrays) so long recordings stay under 8K tokens."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != AUDIO_SR:
        mono = resample(mono, sr, AUDIO_SR)
    win = int(window_s * AUDIO_SR)
    segments = meta.get("segments") or []  # optional per-window transcript [{start,end,text}]
    out = []
    for i, start in enumerate(range(0, max(len(mono), 1), win)):
        piece = mono[start : start + win]
        if len(piece) < AUDIO_SR // 2 and i > 0:  # drop sub-0.5 s tails
            break
        t0, t1 = start / AUDIO_SR, min(start + win, len(mono)) / AUDIO_SR
        seg_text = " ".join(s.get("text", "") for s in segments if s.get("start", 0) < t1 and s.get("end", 0) > t0)
        out.append(make(f"{_mmss(t0)}-{_mmss(t1)}", f"{label} {seg_text}".strip(), media=piece))
    return out


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(x) == 0:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _heading(text: str) -> str | None:
    """First markdown heading, else the first short non-empty line (skipping docstring/emphasis lines)."""
    lines = [s.strip() for s in text.splitlines()[:15]]
    for s in lines:
        if s.startswith("#"):
            return s.lstrip("#").strip()[:120] or None
    for s in lines[:5]:
        if s and len(s) <= 120 and not s.startswith(('"""', "'''", "*")):
            return s
    return None


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"
