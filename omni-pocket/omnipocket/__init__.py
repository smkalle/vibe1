"""Omni Pocket: a phone-first Streamlit workbench for LiquidAI LFM3-Omni-2B.

One end-to-end model (text + image + audio in, text/speech out) replaces the
usual ASR -> VLM -> LLM -> TTS chain. Omni Pocket lets you drive it from an
Android phone (Termux or UserLAnd Arch) through several engines:

* ``webgpu``       - inference inside Android Chrome via WebGPU (component tab)
* ``transformers`` - the Python baseline (UserLAnd Arch / desktop / GPU box)
* ``remote``       - any OpenAI-compatible server, e.g. ``python -m omnipocket serve``
                     on a PC with a GPU, reached over LAN/Tailscale
* ``mock``         - deterministic offline engine for demos and tests
"""

__version__ = "0.1.0"
