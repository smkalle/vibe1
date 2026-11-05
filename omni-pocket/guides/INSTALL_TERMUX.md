# Install on Termux (Android)

Tested layout: Termux from **F-Droid** or GitHub releases. The Play Store
build is outdated and its package mirrors are broken. You need Android 7+ for
the UI, and Android 12+ with Chrome 121+ for WebGPU.

## 1. Prepare Termux

```bash
termux-setup-storage            # optional: lets you pick photos from /sdcard
pkg update -y && pkg install -y git
git clone https://github.com/smkalle/vibe1
cd vibe1/omni-pocket
```

## 2. Install

```bash
./scripts/install_termux.sh
```

What it does, and why:

| Step | Why |
|---|---|
| `pkg install python-numpy python-pillow python-pyarrow` | Prebuilt Termux packages. Compiling these with pip on a phone takes hours or fails |
| `python-pandas` (main repo, then `tur-repo`, then pip as a fallback) | Streamlit needs pandas |
| `rust`, `clang`, `ANDROID_API_LEVEL=$(getprop ro.build.version.sdk)` | A couple of Streamlit dependencies (e.g. `rpds-py`) build from source via maturin, which needs the API level |
| `python -m venv --system-site-packages .venv` | The venv reuses the prebuilt numpy, pyarrow and pandas |
| `pip install -r requirements.txt` | streamlit, requests (no torch) |
| `pkg install llama-cpp` | Runs the model on the phone (needs v0.8.0+ for LFM3-Omni; Termux has 0.9.0) |

Expect 10–20 minutes the first time. Keep the phone plugged in. `CARGO_BUILD_JOBS=2` keeps it from throttling.

## 3. Run

```bash
./scripts/run.sh --llama
```

`--llama` starts `llama-server` in the background with
`LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M`. The first run downloads about 1.5 GB
(model + multimodal projector) into `~/.cache/llama.cpp`, so use Wi-Fi. Follow
it with `tail -f omni_data/llama-server.log`. The app then uses the
**llamacpp** engine. Stop the server with `./scripts/llama_server.sh --stop`.

This takes a wake-lock, starts Streamlit on `127.0.0.1:8501`, and opens
Chrome via `termux-open-url`. Always use **`http://localhost:8501`**:
camera, mic and WebGPU are only allowed in a secure context.

## 4. Choose how the model runs

- **llamacpp engine (recommended)**: the model runs on the phone CPU through llama.cpp, as above. Photos, voice and text all work in the Ask tab. *Read answers aloud with → model voice* uses llama-server's `/tts`.
  - Tuning: `OMNI_LLAMA_THREADS=6` (default: cores − 2), `OMNI_LLAMA_CTX=8192` for longer audio, `OMNI_GGUF=LiquidAI/LFM3-Omni-2B-GGUF:Q8_0` for quality if you have RAM.
  - GPU offload (experimental on phones): `pkg install llama-cpp-backend-vulkan` (or `-opencl` for Adreno), then `OMNI_LLAMA_NGL=99 ./scripts/run.sh --llama`. Compare against the CPU in the Benchmarks tab; on many phones the CPU is faster.
- **Live (WebGPU) tab**: *Check WebGPU* and the GPU benchmark work. *Load model* currently fails with `Unsupported model type: lfm3_omni`, because no transformers.js release supports the model yet.
- **Remote engine**: run `./scripts/serve_gpu.sh` on your PC (see DEPLOY.md), then on the phone:
  ```bash
  OMNI_ENGINE=remote OMNI_REMOTE_URL=http://192.168.1.20:8088/v1 OMNI_REMOTE_API_KEY=... ./scripts/run.sh
  ```
- The **transformers** engine is not available on plain Termux, because PyTorch has no Android wheels on PyPI. Use UserLAnd or `proot-distro install archlinux` for that ([INSTALL_USERLAND_ARCH.md](INSTALL_USERLAND_ARCH.md)).

## Optional extras

```bash
pkg install termux-api          # + install the Termux:API app → "termux-tts-speak" voice option
./scripts/termux_boot.sh        # + Termux:Boot app → start Omni Pocket at boot
./scripts/smoke_test.sh         # offline self-test (pip install pytest first for unit tests)
./scripts/update.sh             # git pull + refresh deps
```

Battery: Android may kill Termux in the background. In Android Settings → Apps → Termux → Battery, choose **Unrestricted**.
