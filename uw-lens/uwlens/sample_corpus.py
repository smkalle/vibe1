"""Build the synthetic bilingual demo corpus (md, py, pdf, png, wav, mp4) from corpus_src/.

All content is authored for this repo (CC0). No real patient data. `*.pdf.md` and `*.png.md`
sources are rendered with PyMuPDF's built-in CJK font so Japanese renders without system fonts.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .config import ROOT

SRC = ROOT / "corpus_src"
PAGE_BREAK = "---page---"


def _font(text: str) -> str:
    # The built-in CJK font sets Latin at full width; use Helvetica for pure-ASCII pages.
    return "helv" if text.isascii() else "japan"


def _plain(md: str) -> str:
    lines = []
    for line in md.splitlines():
        s = line.rstrip()
        while s.startswith("#"):
            s = s[1:]
        lines.append(s.strip().replace("**", "").replace("*", "").replace("\u2014", "-").replace("\u2013", "-"))
    return "\n".join(lines).strip()


def render_pdf(src: Path, dst: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    for part in src.read_text(encoding="utf-8").split(PAGE_BREAK):
        page = doc.new_page(width=595, height=842)  # A4
        body = _plain(part)
        rc = page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), body, fontname=_font(body), fontsize=10.5)
        if rc < 0:
            raise ValueError(f"{src.name}: page text overflows by {-rc:.0f}pt; add a {PAGE_BREAK}")
    doc.set_metadata({"title": src.stem.removesuffix(".pdf"), "creator": "UW Lens sample corpus (synthetic)"})
    doc.save(dst)
    doc.close()


def render_text_image(src: Path, dst: Path, width: int = 640, height: int = 260) -> None:
    """Render text as a 'scanned document' PNG (text lives in pixels only, like a real scan).

    The canvas grows in 40pt steps until the text fits.
    """
    import pymupdf

    text = src.read_text(encoding="utf-8").strip().replace("\u2014", "-").replace("\u2013", "-")
    for h in range(height, 2000, 40):
        doc = pymupdf.open()
        page = doc.new_page(width=width, height=h)
        page.draw_rect(page.rect, color=None, fill=(0.98, 0.97, 0.94))
        page.draw_rect(pymupdf.Rect(12, 12, width - 12, h - 12), color=(0.3, 0.3, 0.3), width=0.8)
        rc = page.insert_textbox(pymupdf.Rect(28, 26, width - 28, h - 20), text,
                                 fontname=_font(text), fontsize=10.5, color=(0.1, 0.1, 0.15))
        if rc >= 0:
            page.get_pixmap(dpi=110).save(dst)
            doc.close()
            return
        doc.close()
    raise ValueError(f"{src.name}: text too long for an image")


def render_trend_chart(dst: Path) -> None:
    """A small line chart (HbA1c + systolic BP over time) drawn with PyMuPDF primitives."""
    import pymupdf

    w, h = 640, 360
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    page.insert_text((24, 30), "Case 0417 - HbA1c (%) and home systolic BP (mmHg), 2024-2026 (synthetic)", fontsize=11)
    x0, x1, y0, y1 = 60, 600, 300, 60
    page.draw_line((x0, y0), (x1, y0), color=(0.4, 0.4, 0.4))
    page.draw_line((x0, y0), (x0, y1), color=(0.4, 0.4, 0.4))
    labels = ["2024-11", "2025-05", "2025-11", "2026-04"]
    hba1c = [8.1, 7.4, 7.1, 6.9]
    sbp = [148, 141, 137, 134]

    def pts(vals, lo, hi):
        step = (x1 - x0 - 40) / (len(vals) - 1)
        return [(x0 + 20 + i * step, y0 - (v - lo) / (hi - lo) * (y0 - y1)) for i, v in enumerate(vals)]

    for series, lo, hi, color, dy in ((hba1c, 6.0, 9.0, (0.1, 0.35, 0.75), 16), (sbp, 120, 160, (0.8, 0.3, 0.1), -8)):
        p = pts(series, lo, hi)
        for a, b in zip(p, p[1:]):
            page.draw_line(a, b, color=color, width=2)
        for (x, y), v in zip(p, series):
            page.draw_circle((x, y), 3, color=color, fill=color)
            page.insert_text((x + 5, y + dy), f"{v}", fontsize=9, color=color)
    for (x, _), lab in zip(pts(hba1c, 6, 9), labels):
        page.insert_text((x - 18, y0 + 16), lab, fontsize=9)
    page.insert_text((x0 + 10, 330), "blue: HbA1c %   orange: systolic BP", fontsize=9)
    page.get_pixmap(dpi=110).save(dst)
    doc.close()


def synth_audio(dst: Path, seconds: float = 75.0, sr: int = 16_000) -> None:
    """Placeholder 'interview' audio: speech-like modulated tones. Its meaning comes from the sidecar transcript."""
    rng = np.random.default_rng(417)
    t = np.arange(int(seconds * sr)) / sr
    syllables = (np.sin(2 * np.pi * 3.5 * t) > 0.2).astype(np.float32)
    pauses = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(np.float32)
    pitch = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    voice = np.sin(2 * np.pi * np.cumsum(pitch) / sr) + 0.3 * np.sin(4 * np.pi * np.cumsum(pitch) / sr)
    sig = 0.25 * voice * syllables * pauses + 0.01 * rng.standard_normal(len(t))
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def synth_video(dst: Path, frame_src: Path, workdir: Path) -> bool:
    """4-second MP4 of a rendered CPAP report screen. Needs ffmpeg; returns False if unavailable."""
    if not shutil.which("ffmpeg"):
        return False
    frame = workdir / "_cpap_frame.png"
    frame_src.write_text(
        "CPAP USAGE REPORT (synthetic) - last 90 days\n\nNights with 4h+ use: 73 / 90 (81%)\n"
        "Average use: 6.2 h/night\nResidual AHI: 3.1 events/h\nMask leak (95th pct): 18 L/min - normal\n"
        "Pressure: auto 6-12 cmH2O",
        encoding="utf-8",
    )
    render_text_image(frame_src, frame, width=640, height=360)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(frame), "-t", "4", "-r", "2",
           "-vf", "scale=640:-2,format=yuv420p", "-c:v", "libx264", str(dst)]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    finally:
        frame.unlink(missing_ok=True)
        frame_src.unlink(missing_ok=True)
    return True


def build(out: Path, src: Path = SRC, *, with_media: bool = True) -> dict[str, int]:
    """(Re)create the corpus in `out`. Returns counts by output type."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    def bump(kind):
        counts[kind] = counts.get(kind, 0) + 1

    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.name == "sidecars.json":
            continue
        rel_dir = out / f.parent.relative_to(src)
        rel_dir.mkdir(parents=True, exist_ok=True)
        if f.name.endswith(".pdf.md"):
            render_pdf(f, rel_dir / f.name.removesuffix(".md"))
            bump("pdf")
        elif f.name.endswith(".png.md"):
            if with_media:
                render_text_image(f, rel_dir / f.name.removesuffix(".md"))
                bump("image")
        else:
            shutil.copy2(f, rel_dir / f.name)
            bump(f.suffix.lstrip("."))

    case = out / "case"
    if with_media:
        render_trend_chart(case / "bp_trend_0417.png")
        bump("image")
        synth_audio(case / "teleinterview_0417.wav")
        bump("audio")
        if synth_video(case / "cpap_compliance_0417.mp4", case / "_cpap.txt", case):
            bump("video")
    for side in src.rglob("sidecars.json"):
        target_dir = out / side.parent.relative_to(src)
        for name, meta in json.loads(side.read_text(encoding="utf-8")).items():
            if (target_dir / name).exists():
                (target_dir / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
    return counts
