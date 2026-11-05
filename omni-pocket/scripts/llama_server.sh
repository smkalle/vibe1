#!/usr/bin/env bash
# Run LFM3-Omni-2B on THIS phone with llama.cpp's llama-server (OpenAI-compatible,
# image + audio input). Omni Pocket's "llamacpp" engine talks to it.
#
#   ./scripts/llama_server.sh                 # foreground, 127.0.0.1:8080
#   ./scripts/llama_server.sh --background    # detach, log in omni_data/llama-server.log
#   ./scripts/llama_server.sh --stop
#
# Env: OMNI_GGUF (default LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M), OMNI_LLAMA_PORT (8080),
#      OMNI_LLAMA_CTX (4096), OMNI_LLAMA_THREADS (big cores), OMNI_LLAMA_NGL (0 = CPU),
#      OMNI_LLAMA_ARGS (extra flags). Weights cache: ~/.cache/llama.cpp (~1.5 GB for Q4_K_M + mmproj).
source "$(dirname "$0")/common.sh"
GGUF="${OMNI_GGUF:-LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M}"
PORT="${OMNI_LLAMA_PORT:-8080}"
CTX="${OMNI_LLAMA_CTX:-4096}"
NGL="${OMNI_LLAMA_NGL:-0}"
LOG="$ROOT/omni_data/llama-server.log"
PIDFILE="$ROOT/omni_data/llama-server.pid"
MODE=fg
for a in "$@"; do case "$a" in --background) MODE=bg;; --stop) MODE=stop;; *) die "unknown flag $a";; esac; done
mkdir -p "$ROOT/omni_data"

if [[ "$MODE" == stop ]]; then
  [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" && say "stopped" || warn "not running"
  exit 0
fi

# llama.cpp v0.8.0 (Oct 28 2026) is the first release with LFM3-Omni (PR #31257).
if ! command -v llama-server >/dev/null 2>&1; then
  case "$(detect_env)" in
    termux) say "Installing llama.cpp (pkg install llama-cpp)"; pkg install -y llama-cpp ;;
    *) die "llama-server not found. Install llama.cpp >= v0.8.0 (see guides/INSTALL_USERLAND_ARCH.md section 3b) and put it on PATH" ;;
  esac
fi
VER="$(llama-server --version 2>&1 | grep -oE 'version: [0-9]+|v?[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
say "llama-server ${VER:-(version unknown)} · model $GGUF · http://127.0.0.1:$PORT"

# Default threads: number of cores minus 2 (leave the little cores for Android + Streamlit).
NCPU="$(nproc 2>/dev/null || echo 4)"
THREADS="${OMNI_LLAMA_THREADS:-$(( NCPU > 4 ? NCPU - 2 : NCPU ))}"

ARGS=(-hf "$GGUF" --host 127.0.0.1 --port "$PORT" -c "$CTX" -t "$THREADS" -ngl "$NGL" --reasoning off)
# shellcheck disable=SC2206
[[ -n "${OMNI_LLAMA_ARGS:-}" ]] && ARGS+=(${OMNI_LLAMA_ARGS})

if [[ "$MODE" == bg ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then say "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
  command -v termux-wake-lock >/dev/null && termux-wake-lock || true
  nohup llama-server "${ARGS[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  say "started (pid $!), first run downloads the GGUF - follow with: tail -f $LOG"
  for i in $(seq 1 900); do   # up to 15 min for the first download on mobile data
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then say "ready after ${i}s"; exit 0; fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -20 "$LOG"; die "llama-server exited"; }
    sleep 1
  done
  die "llama-server not ready after 15 min - see $LOG"
fi
exec llama-server "${ARGS[@]}"
