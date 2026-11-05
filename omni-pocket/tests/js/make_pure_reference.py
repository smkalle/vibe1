"""Reference values for the pure helpers of omnipocket/webgpu/lfm3omni.js, from transformers
(no torch needed) and numpy:

* mel filters: transformers.audio_utils.mel_filter_bank (slaney), as LFM3's feature extractor
* log-mel: the LFM3 recipe (right-pad, |STFT|^2 center=False with a periodic Hann window, slaney
  filters, log10 clamped at -12, (x + 4) / 4), written out here in numpy
* image: transformers' Qwen2VLImageProcessor (PIL backend) with LFM3-Omni-2B's processor settings, smart_resize
* prompt: LFM3-Omni-2B's chat_template.jinja rendered by transformers' jinja renderer

    python tests/js/make_pure_reference.py ASSETS_DIR OUT_DIR

ASSETS_DIR holds LFM3-Omni-2B's processor_config.json and chat_template.jinja (data files from the
model repo). Then: OMNI_REF=OUT_DIR node --test tests/js/test_lfm3omni.mjs
"""

import json
import pathlib
import sys

import numpy as np
from PIL import Image
from transformers.audio_utils import mel_filter_bank
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil as Qwen2VLImageProcessor, smart_resize
from transformers.utils.chat_template_utils import _compile_jinja_template

SR, N_MELS, N_FFT, WIN, HOP, FLOOR, OFFSET, DIVISOR, STRIDE = 16000, 128, 400, 400, 160, -12.0, 4.0, 4.0, 8


def num_frames(n: int) -> int:
    padded = max(N_FFT, n + (-(n - N_FFT) % HOP))
    return 1 + (padded - N_FFT) // HOP


def log_mel(wave: np.ndarray, filters: np.ndarray) -> np.ndarray:
    frames = num_frames(len(wave))
    padded = np.zeros((frames - 1) * HOP + N_FFT)
    padded[: len(wave)] = wave
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32).astype(np.float64)
    segments = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(segments * window, axis=-1)) ** 2
    with np.errstate(divide="ignore"):
        feats = np.log10(filters.T.astype(np.float64) @ power.T)
    return ((np.maximum(feats, FLOOR) + OFFSET) / DIVISOR).astype(np.float32)


def main(assets: pathlib.Path, out: pathlib.Path) -> None:
    data = out / "data"
    (out / "model").mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    processor = json.loads((assets / "processor_config.json").read_text())
    (out / "model" / "processor_config.json").write_text(json.dumps(processor))
    rng = np.random.default_rng(0)

    filters = mel_filter_bank(num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=N_MELS, min_frequency=0.0,
                              max_frequency=SR / 2, sampling_rate=SR, norm="slaney", mel_scale="slaney").astype(np.float32)
    filters.tofile(data / "mel_filters.f32")
    t = np.arange(int(1.3 * SR)) / SR
    wave = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    wave.tofile(data / "wave.f32")
    mel = log_mel(wave, filters)
    mel.tofile(data / "mel.f32")
    n_audio = -(mel.shape[1] // -STRIDE)

    rgb = rng.integers(0, 256, (256, 320, 3), dtype=np.uint8)  # already a smart_resize size: no resampling
    rgb.tofile(data / "image_rgb.u8")
    cfg = {k: v for k, v in processor["image_processor"].items() if k not in ("image_processor_type", "auto_map")}
    out_img = Qwen2VLImageProcessor(**cfg)(images=[Image.fromarray(rgb)], return_tensors="np")
    out_img["pixel_values"].astype(np.float32).tofile(data / "pixel_values.f32")
    grid = out_img["image_grid_thw"][0].tolist()

    sizes = [(480, 640), (1080, 1920), (3000, 4000), (100, 100), (720, 1280), (333, 777), (4032, 3024), (240, 400), (272, 400)]
    resize = [[h, w, cap, *smart_resize(h, w, 32, 65536, cap)] for h, w in sizes for cap in (65536, 256 * 1024, 2097152)]

    template = _compile_jinja_template((assets / "chat_template.jinja").read_text())
    special = {"bos_token": "<|startoftext|>", "eos_token": "<|im_end|>"}
    convs = {
        "full": [{"role": "system", "content": "S"},
                 {"role": "user", "content": [{"type": "image"}, {"type": "audio"}, {"type": "text", "text": "Q"}]}],
        "text_only": [{"role": "user", "content": [{"type": "text", "text": "Hi there"}]}],
        "history": [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "c"}]}],
    }
    image_tokens = grid[0] * grid[1] * grid[2] // 4
    prompt = [124894, 11, 12, 125009, 125030, 125010, 125009, 125031, 125010, 13]
    ref = {
        "mel_frames": int(mel.shape[1]), "n_audio": n_audio, "grid": grid, "smart_resize": resize,
        "templates": {k: template.render(messages=v, add_generation_prompt=True, **special) for k, v in convs.items()},
        "prompt": prompt,
        "ids": [124894, 11, 12, 125009, *[125030] * image_tokens, 125010, 125009, *[125031] * n_audio, 125010, 13],
    }
    (out / "reference.json").write_text(json.dumps(ref, indent=1))
    print(f"grid {grid}, mel {mel.shape}, audio tokens {n_audio}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
