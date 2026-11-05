"""Python baseline: Hugging Face Transformers, exactly the interface from the
model card (``AutoModelForMultimodalLM`` + ``AutoProcessor`` + the processor's
``audio_tokenizer`` codec).

Runs on a desktop/GPU box, and on a phone inside UserLAnd/proot Arch (CPU,
bfloat16, ~4.4 GB of weights). Not on plain Termux: PyTorch has no Android
wheels on PyPI.

If a future model release renames anything, this file is the only place that
needs to change.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

from ..media import to_wav_bytes
from .base import OmniRequest, OmniResult, StreamTimer


def pick_device_dtype(device: str = "auto", dtype: str = "auto"):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype == "auto":
        # The model card uses float32 on CPU. On a phone that is ~8.7 GB, so we
        # default to bfloat16 on CPU too (half the RAM; aarch64 runs it fine).
        dtype = "bfloat16"
    return device, getattr(torch, dtype)


class TransformersEngine:
    name = "transformers"

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto"):
        self.model = model_id
        self.device_pref, self.dtype_pref = device, dtype
        self._loaded = False
        self._lock = threading.Lock()
        self.load_s: float | None = None

    # -- loading ---------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            import torch  # noqa: F401  (fail early with a clear ImportError)
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self.device, self.dtype = pick_device_dtype(self.device_pref, self.dtype_pref)
            self.hf_model = AutoModelForMultimodalLM.from_pretrained(
                self.model, dtype=self.dtype, device_map=self.device, trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
            codec = getattr(self.processor, "audio_tokenizer", None)
            self.codec = codec.to(device=self.device).eval() if codec is not None else None
            self.load_s = time.perf_counter() - t0
            self._loaded = True

    # -- prompt ------------------------------------------------------------
    def _conversation(self, req: OmniRequest, tmp: Path) -> list[dict]:
        conv: list[dict] = [{"role": "system", "content": req.system}]
        for turn in req.history:
            conv.append({"role": turn["role"], "content": [{"type": "text", "text": turn["content"]}]})
        content: list[dict] = []
        if req.image is not None:
            p = tmp / "image.jpg"
            req.image.save(p, format="JPEG", quality=90)
            content.append({"type": "image", "image": str(p)})
        if req.audio is not None:
            p = tmp / "audio.wav"
            p.write_bytes(to_wav_bytes(req.audio))
            content.append({"type": "audio", "audio": str(p)})
        if req.text.strip():
            content.append({"type": "text", "text": req.text.strip()})
        conv.append({"role": "user", "content": content})
        return conv

    def _inputs(self, req: OmniRequest, tmp: Path):
        conv = self._conversation(req, tmp)
        return self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.hf_model.device)

    def _gen_kwargs(self, req: OmniRequest) -> dict:
        kw = {"max_new_tokens": req.max_new_tokens, "generation_mode": "text"}
        if req.temperature > 0:
            kw.update(do_sample=True, temperature=req.temperature)
        else:
            kw.update(do_sample=False)
        return kw

    # -- inference ---------------------------------------------------------
    def stream(self, req: OmniRequest) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        self.load()
        with tempfile.TemporaryDirectory() as td:
            inputs = self._inputs(req, Path(td))
            streamer = TextIteratorStreamer(self.processor.tokenizer if hasattr(self.processor, "tokenizer")
                                            else self.processor, skip_prompt=True, skip_special_tokens=True)
            errors: list[BaseException] = []

            def run():
                try:
                    self.hf_model.generate(**inputs, **self._gen_kwargs(req), streamer=streamer)
                except BaseException as exc:  # surface errors from the worker thread
                    errors.append(exc)
                    streamer.end()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            for piece in streamer:
                if piece:
                    yield piece
            t.join()
            if errors:
                raise errors[0]

    def generate(self, req: OmniRequest) -> OmniResult:
        self.load()
        timer = StreamTimer(self.stream(req))
        for _ in timer:
            pass
        text = timer.text.strip()
        out_tokens = len(self.processor.tokenizer(text)["input_ids"]) if hasattr(self.processor, "tokenizer") else timer.chunks
        result = OmniResult(text=text, engine=self.name, model=self.model, output_tokens=out_tokens,
                            ttft_s=timer.ttft, total_s=timer.elapsed)
        if self.load_s is not None:
            result.notes.append(f"model load {self.load_s:.1f}s ({self.device}, {str(self.dtype).replace('torch.', '')})")
        if req.want_speech:
            audio, note = self.try_speech(req)
            result.audio_wav = audio
            if note:
                result.notes.append(note)
        return result

    def try_speech(self, req: OmniRequest) -> tuple[bytes | None, str | None]:
        """EXPERIMENTAL model-native speech output.

        The model emits interleaved text + audio-codec tokens; the processor's
        ``audio_tokenizer`` decodes the codes to a waveform. The exact generate
        flag / output fields are taken from the model card's interleaved
        example; if they differ in your installed version this returns a note
        instead of crashing, and the UI falls back to on-device TTS.
        """
        import numpy as np
        import torch

        if self.codec is None:
            return None, "speech: processor has no audio_tokenizer"
        try:
            with tempfile.TemporaryDirectory() as td:
                inputs = self._inputs(req, Path(td))
                kw = self._gen_kwargs(req) | {"generation_mode": "interleaved"}
                with torch.inference_mode():
                    out = self.hf_model.generate(**inputs, **kw)
                codes = getattr(out, "audio_codes", None)
                if codes is None:
                    return None, "speech: generate() returned no audio_codes"
                wav = self.codec.decode(codes)
                wav = getattr(wav, "audio_values", wav)
                arr = wav.detach().float().cpu().numpy().reshape(-1)
                sr = int(getattr(self.codec.config, "sampling_rate", 24_000))
                from ..media import Audio

                return to_wav_bytes(Audio(arr.astype(np.float32), sr)), None
        except Exception as exc:
            return None, f"speech unavailable ({exc.__class__.__name__}: {str(exc)[:120]}) - using TTS fallback"
