#!/usr/bin/env bash
# Start the Streamlit UI.
#   ./scripts/run.sh            # localhost only (camera/mic/WebGPU work in Chrome on this phone)
#   ./scripts/run.sh --lan      # also reachable from other devices (no camera/WebGPU there: not https)
#   ./scripts/run.sh --llama    # also start llama-server with LFM3-Omni GGUF on this phone
#   OMNI_ENGINE=remote OMNI_REMOTE_URL=http://pc:8088/v1 ./scripts/run.sh
source "$(dirname "$0")/common.sh"
HOST="${OMNI_HOST:-127.0.0.1}"
PORT="${OMNI_PORT:-8501}"
OPEN=1
LLAMA=0
for a in "$@"; do case "$a" in --lan) HOST=0.0.0.0;; --no-open) OPEN=0;; --llama) LLAMA=1;; *) die "unknown flag $a";; esac; done
if [[ "$LLAMA" == 1 ]]; then
  "$(dirname "$0")/llama_server.sh" --background
  export OMNI_ENGINE=llamacpp
fi

activate
cd "$ROOT"
ENV_KIND="$(detect_env)"
say "Environment: $ENV_KIND  ·  engine: ${OMNI_ENGINE:-auto}  ·  http://localhost:$PORT"

if [[ "$ENV_KIND" == termux ]] && command -v termux-wake-lock >/dev/null; then
  termux-wake-lock || true   # keep Android from killing the server while Chrome is in front
  trap 'termux-wake-unlock 2>/dev/null || true' EXIT
fi
if [[ "$OPEN" == 1 ]]; then
  ( sleep 4
    if command -v termux-open-url >/dev/null; then termux-open-url "http://localhost:$PORT"
    elif command -v xdg-open >/dev/null && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then xdg-open "http://localhost:$PORT"
    else echo "Open http://localhost:$PORT in Chrome"; fi ) >/dev/null 2>&1 &
fi
if [[ "$HOST" == 0.0.0.0 ]]; then
  IP="$(ip -4 addr 2>/dev/null | awk '/inet / && $2 !~ /^127/{sub(/\/.*/,"",$2);print $2;exit}')"
  warn "LAN mode: http://${IP:-<phone-ip>}:$PORT - camera/mic/WebGPU need https or localhost on the viewing device"
fi
exec streamlit run app.py --server.address "$HOST" --server.port "$PORT"
