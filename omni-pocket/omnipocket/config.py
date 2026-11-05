"""Settings, read from environment variables (``OMNI_*``) with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "LiquidAI/LFM3-Omni-2B"
DEFAULT_ONNX_MODEL_ID = "LiquidAI/LFM3-Omni-2B-ONNX"
OFFICIAL_WEBGPU_DEMO = "https://huggingface.co/spaces/LiquidAI/LFM3-Omni-2B-WebGPU"

ENGINES = ("mock", "remote", "transformers")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass
class Settings:
    engine: str = "auto"
    model_id: str = DEFAULT_MODEL_ID
    onnx_model_id: str = DEFAULT_ONNX_MODEL_ID
    remote_url: str = "http://127.0.0.1:8088/v1"
    remote_model: str = DEFAULT_MODEL_ID
    remote_api_key: str = field(default="", repr=False)  # never printed
    device: str = "auto"  # auto | cpu | cuda | mps
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    max_new_tokens: int = 256
    data_dir: Path = Path("omni_data")
    # transformers.js build for the in-browser engine. Pin the version the
    # ONNX model card asks for; it is editable in the sidebar too.
    transformers_js_version: str = "4.0.0"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            engine=_env("OMNI_ENGINE", "auto"),
            model_id=_env("OMNI_MODEL_ID", DEFAULT_MODEL_ID),
            onnx_model_id=_env("OMNI_ONNX_MODEL_ID", DEFAULT_ONNX_MODEL_ID),
            remote_url=_env("OMNI_REMOTE_URL", "http://127.0.0.1:8088/v1"),
            remote_model=_env("OMNI_REMOTE_MODEL", DEFAULT_MODEL_ID),
            remote_api_key=os.environ.get("OMNI_REMOTE_API_KEY", ""),
            device=_env("OMNI_DEVICE", "auto"),
            dtype=_env("OMNI_DTYPE", "auto"),
            max_new_tokens=int(_env("OMNI_MAX_NEW_TOKENS", "256")),
            data_dir=Path(_env("OMNI_DATA_DIR", "omni_data")),
            transformers_js_version=_env("OMNI_TRANSFORMERS_JS", "4.0.0"),
        )

    def resolve_engine(self) -> str:
        """``auto`` -> transformers if torch+transformers import, else remote if
        a URL was configured explicitly, else mock (so the UI always works)."""
        if self.engine != "auto":
            return self.engine
        from .platform import has_module

        if has_module("torch") and has_module("transformers"):
            return "transformers"
        if os.environ.get("OMNI_REMOTE_URL"):
            return "remote"
        return "mock"
