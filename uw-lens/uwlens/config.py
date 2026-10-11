"""Runtime settings. Everything is overridable through UWLENS_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MATRYOSHKA_DIMS = (128, 256, 512, 768)
MODALITY_PROFILES = ("text", "vision", "full")  # 270M / 440M / 740M parameter loads


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # "auto" tries EmbeddingGemma 2 and falls back to the offline hashing embedder.
    embedder: str = field(default_factory=lambda: _env("UWLENS_EMBEDDER", "auto"))
    model_id: str = field(default_factory=lambda: _env("UWLENS_MODEL_ID", "google/embeddinggemma-2"))
    modalities: str = field(default_factory=lambda: _env("UWLENS_MODALITIES", "full"))
    dim: int = field(default_factory=lambda: int(_env("UWLENS_DIM", "768")))
    store: str = field(default_factory=lambda: _env("UWLENS_STORE", "numpy"))
    index_dir: Path = field(default_factory=lambda: Path(_env("UWLENS_INDEX_DIR", str(ROOT / "uwlens_index"))))
    corpus_dir: Path = field(default_factory=lambda: Path(_env("UWLENS_CORPUS_DIR", str(ROOT / "sample_corpus"))))
    redact_phi: bool = field(default_factory=lambda: _env("UWLENS_REDACT_PHI", "1") != "0")
    chunk_chars: int = field(default_factory=lambda: int(_env("UWLENS_CHUNK_CHARS", "1200")))
    audio_window_s: float = field(default_factory=lambda: float(_env("UWLENS_AUDIO_WINDOW_S", "30")))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "gemma3:4b"))
    # Hugging Face access token for the gated model download. Never read from or written to disk by
    # UW Lens (huggingface_hub still honours HF_TOKEN / `huggingface-cli login` on its own); kept out
    # of repr and equality so it can't leak into logs or cache keys.
    hf_token: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.dim not in MATRYOSHKA_DIMS:
            raise ValueError(f"UWLENS_DIM must be one of {MATRYOSHKA_DIMS}, got {self.dim}")
        if self.modalities not in MODALITY_PROFILES:
            raise ValueError(f"UWLENS_MODALITIES must be one of {MODALITY_PROFILES}, got {self.modalities!r}")
        if self.embedder not in ("auto", "gemma", "hashing"):
            raise ValueError(f"UWLENS_EMBEDDER must be auto|gemma|hashing, got {self.embedder!r}")
        if self.store not in ("numpy", "chroma"):
            raise ValueError(f"UWLENS_STORE must be numpy|chroma, got {self.store!r}")

    def with_(self, **kw) -> "Settings":
        return replace(self, **kw)
