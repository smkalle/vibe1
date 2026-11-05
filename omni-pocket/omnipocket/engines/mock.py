"""Offline engine: describes what it received. Lets you exercise the whole UI,
the history log and the benchmark charts on a phone with no model at all."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .base import OmniRequest, OmniResult, StreamTimer


class MockEngine:
    name = "mock"
    model = "mock-omni"

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def _answer(self, req: OmniRequest) -> str:
        parts = []
        if req.image is not None:
            w, h = req.image.size
            arr = np.asarray(req.image.convert("RGB"), dtype=np.float32)
            r, g, b = (int(c) for c in arr.reshape(-1, 3).mean(axis=0))
            bright = "bright" if arr.mean() > 127 else "dark"
            parts.append(f"I see a {w}x{h} {bright} image (mean RGB {r},{g},{b}).")
        if req.audio is not None:
            rms = float(np.sqrt(np.mean(req.audio.samples ** 2))) if len(req.audio.samples) else 0.0
            parts.append(f"I heard {req.audio.seconds:.1f}s of audio (RMS {rms:.3f}).")
        if req.text.strip():
            parts.append(f'You asked: "{req.text.strip()}".')
        if req.history:
            parts.append(f"({len(req.history)} earlier turns in context.)")
        parts.append("[mock engine - choose 'transformers', 'remote' or the WebGPU tab for real answers]")
        return " ".join(parts)

    def stream(self, req: OmniRequest) -> Iterator[str]:
        for word in self._answer(req).split(" "):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield word + " "

    def generate(self, req: OmniRequest) -> OmniResult:
        timer = StreamTimer(self.stream(req))
        for _ in timer:
            pass
        return OmniResult(
            text=timer.text.strip(),
            engine=self.name,
            model=self.model,
            output_tokens=timer.chunks,
            ttft_s=timer.ttft,
            total_s=timer.elapsed,
        )
