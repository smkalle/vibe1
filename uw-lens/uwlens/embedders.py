"""Embedding backends.

GemmaEmbedder wraps `google/embeddinggemma-2` through sentence-transformers and follows the
model card's contract: asymmetric task prompts for text, prompt-free dicts for media,
normalized vectors, optional Matryoshka truncation.

HashingEmbedder is a deterministic, dependency-free fallback (character n-gram feature
hashing). It is lexical, not semantic: it lets the whole app run offline (CI, air-gapped
demos) but cannot see pixels, hear audio, or match across languages. For media it embeds
the sidecar caption/transcript instead.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .config import Settings
from .textproc import normalize, tokenize

log = logging.getLogger(__name__)

MEDIA_KINDS = ("image", "audio", "video")
_PROFILE_KINDS = {"text": (), "vision": ("image", "video"), "full": ("image", "audio", "video")}


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        n = np.linalg.norm(mat)
        return mat / n if n > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def truncate(mat: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation; always re-normalize after cutting."""
    return l2_normalize(np.asarray(mat)[..., :dim])


class Embedder(Protocol):
    id: str  # stable identity recorded in the index manifest
    label: str
    dim: int
    media_kinds: tuple[str, ...]  # media the backend can embed natively

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_documents(self, texts: Sequence[str], code: bool = False) -> np.ndarray: ...
    def embed_media(self, kind: str, payload: Any, fallback_text: str = "") -> np.ndarray: ...


@dataclass
class HashingEmbedder:
    dim: int = 768
    id: str = ""
    label: str = "Offline hashing embedder (lexical fallback)"
    media_kinds: tuple[str, ...] = ()

    def __post_init__(self):
        self.id = f"hashing-v1-{self.dim}"

    def _features(self, text: str) -> list[str]:
        text = normalize(text).lower()
        feats = tokenize(text)
        # character trigrams inside latin words add robustness to inflection (diabetes/diabetic)
        for word in [t for t in feats if t.isascii() and len(t) > 3]:
            padded = f"<{word}>"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return feats

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in self._features(text):
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        # sublinear term frequency, then L2
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        return l2_normalize(vec)

    def embed_queries(self, texts):
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_documents(self, texts, code: bool = False):
        return self.embed_queries(texts)

    def embed_media(self, kind, payload, fallback_text=""):
        return self._embed(fallback_text or "")


class GemmaEmbedder:
    """EmbeddingGemma 2 via sentence-transformers (>=6.1)."""

    def __init__(self, settings: Settings, model: Any | None = None):
        self.dim = settings.dim
        self.model_id = settings.model_id
        self.profile = settings.modalities
        self.media_kinds = _PROFILE_KINDS[self.profile]
        self.model = model if model is not None else self._load(settings)
        prompts = getattr(self.model, "prompts", {}) or {}
        self.query_prompt = _pick(prompts, ("SearchQuery", "query"))
        self.doc_prompt = _pick(prompts, ("Document", "document"))
        self.code_prompt = _pick(prompts, ("CodeRetrieval",)) or self.doc_prompt
        self.id = f"{self.model_id}:{self.dim}"  # all modality profiles share one space
        self.label = f"EmbeddingGemma 2 ({self.profile}, {self.dim}-d)"

    @staticmethod
    def _load(settings: Settings):
        import torch  # noqa: WPS433 - heavy optional deps imported lazily
        from sentence_transformers import SentenceTransformer

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        # never float16: the model card warns it can produce NaNs
        model_kwargs = {"torch_dtype": torch.bfloat16 if use_bf16 else torch.float32}
        config_kwargs = {}
        if settings.modalities == "text":
            config_kwargs = {"vision_config": None, "audio_config": None}
        elif settings.modalities == "vision":
            config_kwargs = {"audio_config": None}
        log.info("loading %s (%s)", settings.model_id, settings.modalities)
        return SentenceTransformer(settings.model_id, model_kwargs=model_kwargs, config_kwargs=config_kwargs)

    def _encode(self, inputs, prompt_name=None) -> np.ndarray:
        kwargs = {"normalize_embeddings": True, "convert_to_numpy": True}
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        if self.dim < 768:
            kwargs["truncate_dim"] = self.dim
        out = np.asarray(self.model.encode(inputs, **kwargs), dtype=np.float32)
        if out.ndim == 1:
            out = out[None, :]
        out = truncate(out, self.dim)  # idempotent; guarantees unit norm after truncation
        if not np.isfinite(out).all():
            raise ValueError("embedding contains NaN/inf (check dtype; avoid float16)")
        return out

    def embed_queries(self, texts):
        return self._encode(list(texts), self.query_prompt)

    def embed_documents(self, texts, code: bool = False):
        return self._encode(list(texts), self.code_prompt if code else self.doc_prompt)

    def embed_media(self, kind, payload, fallback_text=""):
        if kind not in self.media_kinds:
            # encoder not loaded for this profile: embed the caption/transcript as a document
            return self._encode([fallback_text or ""], self.doc_prompt)[0]
        return self._encode([{kind: payload}])[0]


def _pick(prompts: dict, names: Sequence[str]) -> str | None:
    for name in names:
        if name in prompts:
            return name
    return None


def load_embedder(settings: Settings) -> tuple[Embedder, str]:
    """Return (embedder, status message). `auto` falls back to hashing if Gemma can't load."""
    if settings.embedder == "hashing":
        return HashingEmbedder(settings.dim), "Using the offline hashing embedder (UWLENS_EMBEDDER=hashing)."
    try:
        emb = GemmaEmbedder(settings)
        return emb, f"Loaded {emb.label}."
    except Exception as exc:  # ImportError, network, auth, OOM ...
        if settings.embedder == "gemma":
            raise
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        log.warning("EmbeddingGemma 2 unavailable, falling back to hashing: %s", reason)
        return (
            HashingEmbedder(settings.dim),
            f"EmbeddingGemma 2 unavailable ({reason}). Using the offline hashing fallback — "
            "lexical only, no cross-modal or cross-lingual matching.",
        )
