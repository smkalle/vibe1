# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install streamlit` tries to build **pyarrow** on Termux and fails | venv can't see the Termux package | Use `install_termux.sh`. It runs `pkg install python-pyarrow` and creates the venv with `--system-site-packages` |
| `maturin` / `rpds-py` build error mentioning `ANDROID_API_LEVEL` | Rust wheels need the API level on Termux | `export ANDROID_API_LEVEL=$(getprop ro.build.version.sdk)` then rerun pip |
| `No module named torch` with the transformers engine on Termux | No Android PyTorch wheels | Use the **llamacpp** engine (`./scripts/run.sh --llama`) or the remote engine, or UserLAnd/proot Arch |
| Camera/mic buttons do nothing, or "secure context: NO" | Opened via LAN IP over http | Open `http://localhost:8501` on the phone, or use HTTPS (`tailscale serve`) |
| "WebGPU NOT available" | Old Chrome/Android or blocklisted GPU | Update Chrome (121+), Android 12+. Try `chrome://flags/#enable-unsafe-webgpu`. Otherwise use remote |
| `shader-f16: no` | GPU lacks f16 | Set the sidebar dtype to `"q4"` (or a JSON map without `q4f16`) |
| Live tab: `Unsupported model type: lfm3_omni` | No transformers.js release supports LFM3-Omni yet (checked 4.3.2 and 4.4.0-next.0) | Use the **llamacpp** engine: `./scripts/run.sh --llama`. Re-try the Live tab when transformers.js adds `lfm3_omni` (set the version in the sidebar) |
| `llama-server` error: unknown model architecture `lfm3...` | llama.cpp older than v0.8.0 | `pkg upgrade llama-cpp` (Termux), or rebuild from master |
| llamacpp: `image input is not supported - hint: ... mmproj` | Projector not loaded | Use `-hf LiquidAI/LFM3-Omni-2B-GGUF:<quant>` (it fetches the mmproj automatically), or pass `OMNI_LLAMA_ARGS="--mmproj /path/mmproj-LFM3-Omni-2B-Q8_0.gguf"` |
| llamacpp is slow or the phone gets hot | Too many threads on little cores | `OMNI_LLAMA_THREADS=4`; use Q4_K_M; keep audio short |
| Live tab: "this transformers.js build has no multimodal model class" or a load error | transformers.js version doesn't support LFM3-Omni yet, or graph names differ | Set the version, model class and dtype map from the ONNX model card in the sidebar. Or use the official demo button |
| Live tab: out-of-memory / device lost during long questions | Prefill exceeded mobile WebGPU buffer limits | Shorter audio (< 30 s), keep "include camera frame" at 640 px, close other tabs |
| Termux server dies when Chrome is in front | Android battery management | `run.sh` takes a wake-lock. Also set Termux battery usage to **Unrestricted** |
| transformers engine killed during load (UserLAnd) | Not enough RAM | Use bfloat16 (default), close apps, or use the remote engine. Check `python -m omnipocket doctor` |
| `remote error 401` | API key mismatch | Same key in `OMNI_SERVER_API_KEY` (server) and the sidebar / `OMNI_REMOTE_API_KEY` (phone) |
| `Connection refused` to the server | Bound to localhost or firewall | `serve_gpu.sh` binds 0.0.0.0 by default. Open port 8088, or use Tailscale |
| "model voice" gives text only + "speech unavailable" note | Interleaved generate API differs in your transformers version | Expected fallback: the phone TTS reads the answer. Adjust `try_speech()` in `transformers_engine.py` to the model card's speech example |
| `$` amounts look like math in answers | Streamlit LaTeX | Already escaped by `md()` in `app.py`. Report it if you see it |
