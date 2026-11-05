# 🎙️ Omni Pocket — LFM3-Omni-2B on your Android phone

A Streamlit workbench that runs on the phone itself (Termux or UserLAnd Arch) and
drives **LiquidAI LFM3-Omni-2B**. This is one ~2.17B end-to-end model that takes
**text + image + audio in** and gives **text (and speech) out**. It replaces the
usual ASR → VLM → LLM → TTS chain with a single model, fully local.

| Ask tab (phone) | Live WebGPU tab (phone) |
|---|---|
| ![ask](screenshots/ask.png) | ![live](screenshots/live.png) |

## What you can do with it

- **Spoken visual Q&A**: snap a photo, hold the mic, ask "is this ripe?", and hear the answer.
- **Read text, labels and receipts**: transcribe the text plus a one-line summary (total, expiry, dosage).
- **Translate a sign or menu** into any language.
- **Describe my surroundings**: short accessibility descriptions, read aloud.
- **Voice note → clean notes**: transcription rewritten as bullets with action items.
- **"What's wrong here?" / "What can I cook?"**: photo-based diagnosis and recipes.
- **Live camera + voice agent** running *inside Chrome on the phone GPU* (WebGPU).
- **Benchmarks**: every answer logs TTFT and tokens/s per engine and modality, with charts and JSONL export.
- **Device check**: RAM/disk/arch, WebGPU adapter, `shader-f16`, buffer limits, GPU matmul GFLOP/s.

Everything stays on your devices. Logs live in `omni_data/` and media is never stored.

## Pick an engine

| You have | Engine | Where the model runs |
|---|---|---|
| **Termux** (recommended) | `llamacpp`: `./scripts/run.sh --llama` | Phone CPU, llama.cpp + official GGUF Q4_K_M (~1.5 GB) |
| Termux + Chrome 121+ with WebGPU | **Live (WebGPU)** tab | Phone GPU in the browser: LiquidAI's ONNX export (q4f16/q4) on onnxruntime-web, through Omni Pocket's port of their runtime |
| Termux + a **PC with a GPU** | `remote` | PC, via `python -m omnipocket serve` (OpenAI-compatible) |
| **UserLAnd / proot Arch**, ≥ 8 GB RAM | `transformers` | Phone CPU, bfloat16 (slow but fully local Python) |
| Just exploring | `mock` | Nowhere (offline echo; exercises the whole UI) |

llama.cpp has supported LFM3-Omni since **v0.8.0** (PR ggml-org/llama.cpp#31257,
Oct 2026), and Termux ships it as `pkg install llama-cpp` (0.9.0 as of Nov 2026).
`llama-server` accepts the same OpenAI `image_url` and `input_audio` parts this
app sends, and its `/tts` endpoint provides model speech. PyTorch has no wheels
for Android/Bionic, so the `transformers` engine needs a glibc proot such as
UserLAnd.

## Quick start

```bash
# Termux (F-Droid build)
pkg install -y git && git clone https://github.com/smkalle/vibe1 && cd vibe1/omni-pocket
./scripts/install_termux.sh          # 10-20 min the first time (compiles a few wheels), installs llama-cpp
./scripts/run.sh --llama             # starts llama-server (first run downloads the GGUF) + opens http://localhost:8501

# UserLAnd / proot-distro Arch Linux ARM
sudo pacman -S --needed git && git clone https://github.com/smkalle/vibe1 && cd vibe1/omni-pocket
./scripts/install_userland_arch.sh --local   # adds torch + transformers
./scripts/run.sh

# GPU box serving the phone
./scripts/install_desktop.sh --server
OMNI_SERVER_API_KEY=$(openssl rand -hex 16) ./scripts/serve_gpu.sh
# phone: OMNI_ENGINE=remote OMNI_REMOTE_URL=http://<pc>:8088/v1 OMNI_REMOTE_API_KEY=<key> ./scripts/run.sh
```

Then check it all works offline: `./scripts/smoke_test.sh` (pytest, mock CLI, and a headless Streamlit boot).

## Guides

| Guide | Contents |
|---|---|
| [guides/TUTORIAL.md](guides/TUTORIAL.md) | The full tutorial: why, the Python baseline, the browser/WebGPU path, the native/mobile path, the real-time agent, optimisation |
| [guides/INSTALL_TERMUX.md](guides/INSTALL_TERMUX.md) | Termux step by step, Termux:API voice, Termux:Boot autostart |
| [guides/INSTALL_USERLAND_ARCH.md](guides/INSTALL_USERLAND_ARCH.md) | UserLAnd / proot-distro Arch, the local transformers engine, RAM budget |
| [guides/DEPLOY.md](guides/DEPLOY.md) | GPU server, Tailscale, systemd, API keys, LAN/HTTPS, updating |
| [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md) | Known failure modes and fixes |

## CLI

```bash
python -m omnipocket doctor                                   # device report + engine recommendation
python -m omnipocket ask --engine remote --image photo.jpg --audio question.wav --stream
python -m omnipocket bench --engine transformers --text "Hello" --runs 5 --warmup 1
python -m omnipocket serve --engine transformers --host 0.0.0.0 --port 8088
```

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `OMNI_ENGINE` | `auto` | `mock`, `llamacpp`, `remote` or `transformers`. `auto` picks llamacpp if a llama-server answers on `OMNI_LLAMA_URL`, else transformers if torch is installed, else remote if `OMNI_REMOTE_URL` is set, else mock |
| `OMNI_LLAMA_URL` | `http://127.0.0.1:8080/v1` | llama-server for the `llamacpp` engine |
| `OMNI_GGUF` · `OMNI_LLAMA_CTX` · `OMNI_LLAMA_THREADS` · `OMNI_LLAMA_NGL` · `OMNI_LLAMA_ARGS` | `LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M` · 4096 · cores−2 · 0 · | `scripts/llama_server.sh` settings |
| `OMNI_MODEL_ID` | `LiquidAI/LFM3-Omni-2B` | HF repo for the transformers engine |
| `OMNI_ONNX_MODEL_ID` | `LiquidAI/LFM3-Omni-2B-ONNX` | ONNX repo for the WebGPU tab |
| `OMNI_ORT_WEB` | `1.30.0` | onnxruntime-web version for the WebGPU tab (the one LiquidAI's browser tests pin) |
| `OMNI_TRANSFORMERS_JS` | `4.3.2` | transformers.js version, used only for the tokenizer in the WebGPU tab |
| `OMNI_REMOTE_URL` / `OMNI_REMOTE_MODEL` / `OMNI_REMOTE_API_KEY` | | Remote engine target (the key is never written to disk) |
| `OMNI_DEVICE` / `OMNI_DTYPE` | `auto` | transformers engine (`auto` dtype = bfloat16) |
| `OMNI_MAX_NEW_TOKENS` | `256` | Generation budget |
| `OMNI_DATA_DIR` | `omni_data` | History and benchmark JSONL |
| `OMNI_HOST` / `OMNI_PORT` | `127.0.0.1` / `8501` | UI bind address (`run.sh --lan` = `0.0.0.0`) |
| `OMNI_SERVER_API_KEY` | | Require `Bearer` auth on `serve` |

## Layout

```
app.py                       Streamlit UI (Ask · Live (WebGPU) · Benchmarks · Device)
omnipocket/
  engines/{mock,remote,transformers_engine}.py   one contract: generate() + stream()
  webgpu/index.html          bidirectional Streamlit component (no npm build): probe, matmul bench, live agent
  webgpu/lfm3omni.js         LFM3-Omni runtime for onnxruntime-web: port of LiquidAI's liquidonnx loop
  serve.py                   OpenAI-compatible FastAPI server around any engine
  media.py                   EXIF-aware image prep; WAV decode/resample to 16 kHz with numpy only
  presets.py · history.py · platform.py · cli.py · config.py
scripts/                     install_{termux,userland_arch,desktop}.sh, run.sh, llama_server.sh, serve_gpu.sh,
                             termux_boot.sh, smoke_test.sh, update.sh
tests/                       45 offline Python tests (engines, server round-trip, media, AppTest UI)
tests/js/                    lfm3omni.js checks: pure helpers vs transformers, toy bundle on ort-web (Node + WebGPU)
```

## Status and honesty notes

- **Verified here**: the full test suite. A real HTTP round trip from phone client → server → engine. The UI on a Pixel-7 viewport in headless Chromium. The WebGPU probe sending results back to Python. The WGSL matmul benchmark.
- **Checked against llama.cpp source** (v0.9.0 / master, Nov 2026): LFM3-Omni support, the `-hf`, `--reasoning off`, `-c`, `-t` and `-ngl` flags, `image_url`/`input_audio` parsing (data URLs or raw base64), `/health` and `/tts`. Termux's `llama-cpp` package is 0.9.0.
- **Not verified here**: running the real model weights. Hugging Face was unreachable from the build sandbox. The transformers engine follows the model card's Python interface as given.
- **WebGPU tab**: transformers.js can't load LiquidAI's ONNX export (`Unsupported model type: lfm3_omni`; their README says the exports are "not loadable by Transformers.js"). So `lfm3omni.js` ports the text-answer path of LiquidAI's own runtime ([Liquid4All/onnx-export](https://github.com/Liquid4All/onnx-export), `liquidonnx.lfm3_omni`) to onnxruntime-web, using the graphs and settings of that README's WebGPU section. Checked here:
  - **Pure parts** match their references exactly or within 1e-4: mel filter bank and log-mel (transformers / numpy), `smart_resize` and patchify (transformers' Qwen2-VL processor with LFM3-Omni-2B's settings), and the prompt (LFM3-Omni-2B's real `chat_template.jinja`).
  - **The decoder loop**: feeds, external weights, cache handoff, placeholder scatter, chunked prefill and suppression. It reproduces a NumPy reference token for token on a toy model with the real decoder's interface, both on onnxruntime-web's wasm backend in Node and on its WebGPU EP in headless Chromium.
  - **Not yet checked**: LiquidAI's real graphs, whether the tiny export or the 2B weights. Hugging Face is unreachable from the build sandbox. `tests/js/make_reference.py` plus `browser_check.mjs` compare against LiquidAI's own Python runtime once you can run their exporter. Speech output (depth decoder + codec) is not ported.
- **Model-native speech output** (`speech: model voice`) is experimental. When it isn't available, answers are spoken by the phone's TTS (browser `speechSynthesis` or `termux-tts-speak`).
