"""Bidirectional Streamlit component for the in-browser (WebGPU) engine.

``index.html`` speaks the Streamlit component postMessage protocol directly,
so there is no npm build step - important on a phone."""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent
_component = None


def webgpu_panel(*, key: str, **args):
    """Render the panel. Returns the latest event dict from the browser
    (``kind`` in probe | gpu_bench | model_loaded | answer | error) or None."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("omni_webgpu", path=str(_DIR))
    return _component(key=key, default=None, **args)
