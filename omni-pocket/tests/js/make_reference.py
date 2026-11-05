"""Build a tiny LFM3-Omni ONNX export and the Python reference outputs that
tests/js/test_lfm3omni.mjs compares the browser runtime (omnipocket/webgpu/lfm3omni.js) with.

Needs a checkout of https://github.com/Liquid4All/onnx-export (LiquidAI's exporter + runtime)
and its dependencies (torch, transformers>=5.17, onnx, onnxruntime>=1.30, onnx-ir, onnxscript):

    python tests/js/make_reference.py /path/to/onnx-export /tmp/omni-ref

Everything is seeded and offline: the tiny checkpoint is random, with LFM3-Omni-2B's layout,
token ids, processor_config.json and chat template.
"""

import json
import pathlib
import shutil
import sys

import numpy as np


def main(repo: pathlib.Path, out: pathlib.Path) -> None:
    sys.path[:0] = [str(repo), str(repo / "src")]
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    from liquidonnx.lfm3_audio.session import log_mel, mel_filters
    from liquidonnx.lfm3_omni import export as omni_export
    from liquidonnx.lfm3_omni.session import ONNXOmniModel
    from liquidonnx.lfm3_vl.session import expand_image_tokens, preprocess_images
    from tests.test_lfm3_audio import tiny as audio_tiny

    if out.exists():
        shutil.rmtree(out)
    scratch = out / "scratch"
    audio_ckpt = audio_tiny.write_tiny_checkpoint(scratch / "lfm3_audio")
    omni_ckpt = audio_tiny.write_omni_checkpoint(audio_ckpt, scratch / "lfm3_omni")
    model_dir = out / "model"
    onnx_dir = model_dir / "onnx"
    omni_export.export_fp32(omni_ckpt, onnx_dir)
    omni_export.derive(onnx_dir, ["q4"], block_size=32)
    omni_export.copy_checkpoint_files(omni_ckpt, model_dir)
    shutil.rmtree(scratch)

    rng = np.random.default_rng(0)
    data = out / "data"
    data.mkdir()

    # image already at a smart_resize size (256x320 -> grid 16x20 -> 80 tokens): no resampling
    rgb = rng.integers(0, 256, (256, 320, 3), dtype=np.uint8)
    rgb.tofile(data / "image_rgb.u8")
    model = ONNXOmniModel(model_dir, "fp32", force_cpu=True)
    model.load_sessions("cpu")
    model.mel_filters = mel_filters(model.config.mel)
    from liquidonnx.lfm3_vl.session import image_processor

    model.image_processor = image_processor(model_dir)
    pixel_values, grids = preprocess_images(model.image_processor, [Image.fromarray(rgb)])
    pixel_values.astype(np.float32).tofile(data / "pixel_values.f32")
    image_features, _ = model.encode_images([Image.fromarray(rgb)])

    # 1.3 s of speech-like audio at 16 kHz
    t = np.arange(int(1.3 * 16000)) / 16000
    wave = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t)
            + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    wave.tofile(data / "wave.f32")
    mel = log_mel(wave, model.config.mel, model.mel_filters)
    mel.astype(np.float32).tofile(data / "mel.f32")
    model.mel_filters.astype(np.float32).tofile(data / "mel_filters.f32")
    audio_features = model.encode_audio([wave])
    n_audio = model.config.mel.num_tokens(model.config.mel.num_frames(len(wave)))

    tok = model.config.tokens
    image_tok = model.vision_tokens.image
    start, end = 125009, 125010
    text = lambda n, s: np.random.default_rng(s).integers(1000, 120000, n).tolist()  # noqa: E731
    prompt = [tok.bos, *text(5, 1), start, image_tok, end, start, tok.audio, end, *text(6, 2)]
    ids = expand_image_tokens(prompt, grids, image_tok, model.vision_config.merge_size)
    ids = [t for x in ids for t in ([x] * n_audio if x == tok.audio else [x])]

    def generate(precision):
        m = ONNXOmniModel(model_dir, precision, force_cpu=True)
        m.load_sessions("cpu")
        gen = m.generate_ids(ids, audio_features, max_new_tokens=24, suppress_mixed=True,
                             suppress_audio_out=True, image_features=image_features)
        return gen.tokens

    sizes = [(480, 640), (1080, 1920), (3000, 4000), (100, 100), (720, 1280), (333, 777), (4032, 3024)]
    caps = [65536, 256 * 1024, 2097152]
    resize = [[h, w, cap, *smart_resize(h, w, 32, 65536, cap)] for h, w in sizes for cap in caps]

    ref = {
        "prompt": prompt,
        "ids": ids,
        "grid": grids[0].tolist(),
        "n_audio": n_audio,
        "mel_frames": int(mel.shape[1]),
        "image_features_shape": list(image_features.shape),
        "audio_features_shape": list(audio_features.shape),
        "image_features_head": image_features.reshape(-1)[:64].tolist(),
        "audio_features_head": audio_features.reshape(-1)[:64].tolist(),
        "tokens_fp32": generate("fp32"),
        "tokens_q4": generate("q4"),
        "smart_resize": resize,
    }
    # the real LFM3-Omni-2B chat template (vendored in onnx-export's test assets), rendered the
    # way transformers renders it, for buildPrompt
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template(
        (repo / "tests/test_lfm3_audio/assets/LFM3-Omni-2B/chat_template.jinja").read_text())
    special = {"bos_token": "<|startoftext|>", "eos_token": "<|im_end|>"}
    convs = {
        "full": [{"role": "system", "content": "S"},
                 {"role": "user", "content": [{"type": "image"}, {"type": "audio"}, {"type": "text", "text": "Q"}]}],
        "text_only": [{"role": "user", "content": [{"type": "text", "text": "Hi there"}]}],
        "history": [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "c"}]}],
    }
    ref["templates"] = {k: template.render(messages=v, add_generation_prompt=True, **special)
                        for k, v in convs.items()}
    image_features.astype(np.float32).tofile(data / "image_features.f32")
    audio_features.astype(np.float32).tofile(data / "audio_features.f32")
    (out / "reference.json").write_text(json.dumps(ref, indent=1))
    print(json.dumps({k: v for k, v in ref.items() if k.startswith("tokens") or k in ("grid", "n_audio")}))
    print("files:", sorted(p.name for p in onnx_dir.iterdir()))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve())
