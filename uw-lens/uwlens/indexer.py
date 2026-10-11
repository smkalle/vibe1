"""Incremental folder indexing: only new/changed files are re-embedded; deleted files are purged."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .embedders import Embedder
from .extractors import Chunk, extract, fingerprint, iter_files
from .store import VectorStore
from .textproc import find_phi

Progress = Callable[[int, int, str], None]


@dataclass
class IndexReport:
    indexed: list[tuple[str, int]] = field(default_factory=list)  # (path, chunks)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    phi_warnings: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def chunks(self) -> int:
        return sum(n for _, n in self.indexed)


def embed_chunks(embedder: Embedder, chunks: list[Chunk], batch: int = 16) -> np.ndarray:
    vecs: list[np.ndarray] = []
    texts = [c for c in chunks if c.modality in ("text", "pdf", "code")]
    for start in range(0, len(texts), batch):
        group = texts[start : start + batch]
        code = [c for c in group if c.modality == "code"]
        prose = [c for c in group if c.modality != "code"]
        if prose:
            vecs.extend(zip(prose, embedder.embed_documents([c.text for c in prose])))
        if code:
            vecs.extend(zip(code, embedder.embed_documents([c.text for c in code], code=True)))
    for c in chunks:
        if c.modality in ("image", "audio", "video"):
            vecs.append((c, embedder.embed_media(c.modality, c.media, fallback_text=c.text)))
    order = {c.id: i for i, c in enumerate(chunks)}
    vecs.sort(key=lambda cv: order[cv[0].id])
    return np.stack([v for _, v in vecs]) if vecs else np.zeros((0, embedder.dim), np.float32)


def index_folder(folder: Path, embedder: Embedder, store: VectorStore, *, redact: bool = True,
                 chunk_chars: int = 1200, audio_window_s: float = 30.0, modalities: set[str] | None = None,
                 progress: Progress | None = None, force: bool = False) -> IndexReport:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    report = IndexReport()
    files = list(iter_files(folder))
    known = store.files()
    present = {str(p) for p in files}

    # purge vectors for files that disappeared from this folder
    gone = [p for p in known if p not in present and Path(p).is_relative_to(folder)]
    if gone:
        store.delete_paths(gone)
        for p in gone:
            store.set_file(p, None)
        report.removed.extend(gone)

    for i, path in enumerate(files):
        sp = str(path)
        if progress:
            progress(i, len(files), path.name)
        try:
            # settings that change stored text/vectors are part of the fingerprint, so toggling
            # redaction (or chunking) re-processes files instead of keeping stale, unredacted chunks
            fp = f"{fingerprint(path)}|redact={int(redact)}|chunk={chunk_chars}|win={audio_window_s:g}"
            if not force and known.get(sp) == fp:
                report.unchanged.append(sp)
                continue
            chunks = extract(path, folder, chunk_chars=chunk_chars, redact=redact, audio_window_s=audio_window_s,
                             load_audio="audio" in getattr(embedder, "media_kinds", ()))
            if modalities is not None:
                chunks = [c for c in chunks if c.modality in modalities]
            store.delete_paths([sp])
            if not chunks:
                store.set_file(sp, None)
                report.skipped.append((sp, "no extractable content" if modalities is None else "modality filtered"))
                continue
            vectors = embed_chunks(embedder, chunks)
            store.add([c.record() for c in chunks], vectors)
            store.set_file(sp, fp)
            report.indexed.append((sp, len(chunks)))
            if redact:
                leaks = sorted({k for c in chunks for k in find_phi(c.text)})
                if leaks:
                    report.phi_warnings.append((sp, leaks))
        except Exception as exc:  # one bad file must not stop the run
            report.skipped.append((sp, f"{type(exc).__name__}: {exc}"[:300]))
    store.save()
    if progress:
        progress(len(files), len(files), "done")
    return report
