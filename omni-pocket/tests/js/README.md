# Tests for the WebGPU runtime (`omnipocket/webgpu/lfm3omni.js`)

`lfm3omni.js` ports the text-answer path of LiquidAI's ONNX runtime (`liquidonnx.lfm3_omni` in
[Liquid4All/onnx-export](https://github.com/Liquid4All/onnx-export)) to onnxruntime-web. These
checks need Node 20+, `npm install onnxruntime-web@1.30.0` somewhere (pass its folder as
`ORT_WEB_DIR`), and Python with `transformers numpy pillow jinja2 onnx onnxruntime` (no torch).

| Check | What it compares | Run |
|---|---|---|
| Pure helpers | mel filters + log-mel, `smart_resize`, patchify, prompt vs transformers / numpy and LFM3-Omni-2B's real `chat_template.jinja` and `processor_config.json` | `python tests/js/make_pure_reference.py ASSETS /tmp/omni-pure` then `OMNI_REF=/tmp/omni-pure node --test tests/js/test_lfm3omni.mjs` |
| Decoder loop (Node, wasm) | greedy tokens on a toy bundle with the real decoder's interface vs a numpy reference, prefill chunks 1024/5/3/1 | `python tests/js/make_toy_model.py /tmp/omni-toy` then `OMNI_TOY=/tmp/omni-toy ORT_WEB_DIR=... node --test tests/js/test_toy_runtime.mjs` |
| Decoder loop (Chromium, WebGPU) | the same on the WebGPU EP: GPU-resident caches, `getData`, dispose | `OMNI_TOY=/tmp/omni-toy ORT_WEB_DIR=... node tests/js/browser_toy.mjs` (needs Playwright) |
| Real LFM3-Omni graphs | a tiny seeded LFM3-Omni export vs LiquidAI's Python runtime (encoders, greedy tokens fp32 + q4) | `python tests/js/make_reference.py /path/to/onnx-export /tmp/omni-ref` (runs LiquidAI's exporter; needs torch) then `OMNI_REF=/tmp/omni-ref ... node --test tests/js/test_lfm3omni.mjs` and `node tests/js/browser_check.mjs` |

`ASSETS` is a folder with LFM3-Omni-2B's `processor_config.json` and `chat_template.jinja` (from the
model repo, or `tests/test_lfm3_audio/assets/LFM3-Omni-2B/` in onnx-export).
