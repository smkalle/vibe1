#!/usr/bin/env bash
# Run the model on this machine and serve it to phones (OpenAI-compatible).
#   OMNI_SERVER_API_KEY=$(openssl rand -hex 16) ./scripts/serve_gpu.sh
#   phone:  OMNI_ENGINE=remote OMNI_REMOTE_URL=http://<this-host>:8088/v1 OMNI_REMOTE_API_KEY=... ./scripts/run.sh
source "$(dirname "$0")/common.sh"
activate
cd "$ROOT"
HOST="${OMNI_SERVE_HOST:-0.0.0.0}"; PORT="${OMNI_SERVE_PORT:-8088}"; ENGINE="${OMNI_SERVE_ENGINE:-transformers}"
[[ -z "${OMNI_SERVER_API_KEY:-}" && "$HOST" != 127.0.0.1 ]] && warn "no OMNI_SERVER_API_KEY set - anyone on your network can use this server"
exec python -m omnipocket serve --engine "$ENGINE" --host "$HOST" --port "$PORT"
