"""Engine registry."""

from __future__ import annotations

from ..config import Settings
from .base import Engine, OmniRequest, OmniResult, StreamTimer
from .mock import MockEngine
from .remote import RemoteEngine

__all__ = ["Engine", "OmniRequest", "OmniResult", "StreamTimer", "make_engine",
           "MockEngine", "RemoteEngine"]


def make_engine(name: str, settings: Settings):
    if name == "mock":
        return MockEngine()
    if name == "remote":
        return RemoteEngine(settings.remote_url, settings.remote_model, settings.remote_api_key)
    if name == "transformers":
        from .transformers_engine import TransformersEngine  # heavy deps, import lazily

        return TransformersEngine(settings.model_id, settings.device, settings.dtype)
    raise ValueError(f"unknown engine {name!r}; choose mock | remote | transformers")
