"""Engine contract shared by every backend."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..media import Audio


@dataclass
class OmniRequest:
    system: str = "You are a helpful on-device assistant. Answer briefly."
    text: str = ""
    image: object | None = None  # PIL.Image (RGB)
    audio: Audio | None = None  # 16 kHz mono
    history: list[dict] = field(default_factory=list)  # [{"role": ..., "content": str}]
    max_new_tokens: int = 256
    temperature: float = 0.3
    want_speech: bool = False  # ask for interleaved text+speech when the engine can

    def modalities(self) -> list[str]:
        mods = []
        if self.text.strip():
            mods.append("text")
        if self.image is not None:
            mods.append("image")
        if self.audio is not None:
            mods.append("audio")
        return mods

    def is_empty(self) -> bool:
        return not self.modalities()


@dataclass
class OmniResult:
    text: str
    engine: str
    model: str
    audio_wav: bytes | None = None  # model-native speech, if produced
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None  # time to first token
    total_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def tokens_per_s(self) -> float | None:
        if not self.output_tokens or not self.total_s:
            return None
        decode_s = self.total_s - (self.ttft_s or 0.0)
        return self.output_tokens / decode_s if decode_s > 0 else None


class Engine(Protocol):
    name: str
    model: str

    def generate(self, req: OmniRequest) -> OmniResult: ...

    def stream(self, req: OmniRequest) -> Iterator[str]: ...


class StreamTimer:
    """Wraps a token iterator, recording TTFT and token count."""

    def __init__(self, it: Iterator[str]):
        self._it = it
        self.start = time.perf_counter()
        self.ttft: float | None = None
        self.chunks = 0
        self.parts: list[str] = []

    def __iter__(self):
        for piece in self._it:
            if self.ttft is None:
                self.ttft = time.perf_counter() - self.start
            self.chunks += 1
            self.parts.append(piece)
            yield piece

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def text(self) -> str:
        return "".join(self.parts)
