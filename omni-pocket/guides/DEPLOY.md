# Deploy: GPU server + phone client

The most responsive setup: the model runs on a PC or laptop with a GPU, and
the phone runs the light Streamlit UI and sends photos and voice to it.

```
Phone (Termux) ── Streamlit :8501 ── remote engine ──HTTP──► PC :8088  omnipocket serve ─► LFM3-Omni-2B (CUDA)
```

## 1. Server (Linux/WSL/macOS with a GPU)

```bash
git clone https://github.com/smkalle/vibe1 && cd vibe1/omni-pocket
./scripts/install_desktop.sh --server
export OMNI_SERVER_API_KEY=$(openssl rand -hex 16); echo "$OMNI_SERVER_API_KEY"
./scripts/serve_gpu.sh                     # 0.0.0.0:8088, engine=transformers
curl -s localhost:8088/health              # {"ok":true,"engine":"transformers",...}
```

API (OpenAI-compatible):

```bash
curl -s localhost:8088/v1/chat/completions -H "Authorization: Bearer $OMNI_SERVER_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in French"}],"max_tokens":32}'
```

Content parts: `{"type":"text"}`, `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`,
`{"type":"input_audio","input_audio":{"data":"<base64 wav>","format":"wav"}}`.
Add `"modalities":["text","audio"]` to request model speech (experimental). Add `"stream":true` for SSE.

## 2. Reaching the server from the phone

- **Same Wi-Fi**: `OMNI_REMOTE_URL=http://<pc-lan-ip>:8088/v1`
- **Anywhere (recommended)**: install **Tailscale** on both the PC and the phone (Android app), then use `http://<pc-tailscale-name>:8088/v1`. Traffic is WireGuard-encrypted and nothing is exposed to the internet.
- Never port-forward the server to the internet. If you must, put it behind HTTPS (Caddy) and keep the API key.

On the phone:

```bash
OMNI_ENGINE=remote OMNI_REMOTE_URL=http://pc.tailnet.ts.net:8088/v1 \
OMNI_REMOTE_API_KEY=<key> ./scripts/run.sh
```

You can also pick **remote** in the sidebar and paste the URL and key there.
The key stays in session memory only.

## 3. systemd service on the server

```ini
# /etc/systemd/system/omni-pocket.service
[Unit]
Description=Omni Pocket LFM3-Omni server
After=network-online.target

[Service]
User=YOU
WorkingDirectory=/home/YOU/vibe1/omni-pocket
Environment=OMNI_SERVER_API_KEY=CHANGE_ME
Environment=HF_HOME=/home/YOU/.cache/huggingface
ExecStart=/home/YOU/vibe1/omni-pocket/scripts/serve_gpu.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now omni-pocket
journalctl -u omni-pocket -f
```

## 4. Phone autostart

Termux: `./scripts/termux_boot.sh` (needs the Termux:Boot app). UserLAnd has no
boot hook; start it with `./scripts/run.sh` in a session.

## 5. Using the UI from another device (LAN mode)

`./scripts/run.sh --lan` binds 0.0.0.0. Text and uploads work from a laptop
browser. **Camera, mic and WebGPU do not**, because browsers only allow them
on `https://` or `localhost`. For that, use an HTTPS reverse proxy such as
`tailscale serve 8501`, which gives you `https://phone.tailnet.ts.net`.

## 6. Updating and verifying

```bash
./scripts/update.sh        # git pull + pip upgrade in the venv
./scripts/smoke_test.sh    # tests, mock CLI, headless UI boot
```
