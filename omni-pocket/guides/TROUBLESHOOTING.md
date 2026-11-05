# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install streamlit` tries to build **pyarrow** on Termux and fails | venv can't see the Termux package | Use `install_termux.sh`. It runs `pkg install python-pyarrow` and creates the venv with `--system-site-packages` |
| `maturin` / `rpds-py` build error mentioning `ANDROID_API_LEVEL` | Rust wheels need the API level on Termux | `export ANDROID_API_LEVEL=$(getprop ro.build.version.sdk)` then rerun pip |
| `No module named torch` with the transformers engine on Termux | No Android PyTorch wheels | Use the **llamacpp** engine (`./scripts/run.sh --llama`) or the remote engine, or UserLAnd/proot Arch |
| Camera/mic buttons do nothing, or "secure context: NO" | Opened via LAN IP over http | Open `http://localhost:8501` on the phone, or use HTTPS (`tailscale serve`) |
| "WebGPU NOT available" | Old Chrome/Android or blocklisted GPU | Update Chrome (121+), Android 12+. Try `chrome://flags/#enable-unsafe-webgpu`. Otherwise use remote |
| `shader-f16: no` | GPU lacks f16 | *Decoder: auto* already switches to `decoder_q4`; nothing to do |
| `Unsupported model type: lfm3_omni` (older Omni Pocket, or other apps) | transformers.js can't load LiquidAI's LFM3 ONNX exports, by design | Update Omni Pocket (`./scripts/update.sh`): the Live tab now runs them with its own onnxruntime-web runtime (`lfm3omni.js`). Or use the **llamacpp** engine |
| Live tab: `Load failed: 404 ... decoder_q4f16.onnx` | The repo's file names differ from LiquidAI's exporter defaults | Pick *Decoder: q4* in the sidebar, or open the repo's `onnx/` folder and report the names |
| Live tab: garbage or empty answers on a non-Apple, non-NVIDIA GPU | onnxruntime-web 1.30.0's subgroup FlashAttention on some adapters (onnx-export README 4.6) | Set *Prefill chunk* lower (256), or try a newer *onnxruntime-web version*; otherwise use llamacpp |
| `llama-server` error: unknown model architecture `lfm3...` | llama.cpp older than v0.8.0 | `pkg upgrade llama-cpp` (Termux), or rebuild from master |
| llamacpp: `image input is not supported - hint: ... mmproj` | Projector not loaded | Use `-hf LiquidAI/LFM3-Omni-2B-GGUF:<quant>` (it fetches the mmproj automatically), or pass `OMNI_LLAMA_ARGS="--mmproj /path/mmproj-LFM3-Omni-2B-Q8_0.gguf"` |
| llamacpp is slow or the phone gets hot | Too many threads on little cores | `OMNI_LLAMA_THREADS=4`; use Q4_K_M; keep audio short |
| Live tab: out-of-memory / device lost during long questions | Prefill exceeded mobile WebGPU buffer limits | Shorter audio (< 30 s), keep "include camera frame" at 640 px, close other tabs |
| Termux server dies when Chrome is in front | Android battery management | `run.sh` takes a wake-lock. Also set Termux battery usage to **Unrestricted** |
| transformers engine killed during load (UserLAnd) | Not enough RAM | Use bfloat16 (default), close apps, or use the remote engine. Check `python -m omnipocket doctor` |
| `remote error 401` | API key mismatch | Same key in `OMNI_SERVER_API_KEY` (server) and the sidebar / `OMNI_REMOTE_API_KEY` (phone) |
| `Connection refused` to the server | Bound to localhost or firewall | `serve_gpu.sh` binds 0.0.0.0 by default. Open port 8088, or use Tailscale |
| "model voice" gives text only + "speech unavailable" note | Interleaved generate API differs in your transformers version | Expected fallback: the phone TTS reads the answer. Adjust `try_speech()` in `transformers_engine.py` to the model card's speech example |
| `$` amounts look like math in answers | Streamlit LaTeX | Already escaped by `md()` in `app.py`. Report it if you see it |
