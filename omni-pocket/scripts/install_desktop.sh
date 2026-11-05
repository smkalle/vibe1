#!/usr/bin/env bash
# Linux / macOS / WSL / GPU box.
#   ./scripts/install_desktop.sh            # UI only
#   ./scripts/install_desktop.sh --local    # + PyTorch baseline (uses CUDA wheels if available)
#   ./scripts/install_desktop.sh --server   # + OpenAI-compatible server for phones
#   ./scripts/install_desktop.sh --dev      # + pytest
source "$(dirname "$0")/common.sh"
REQ=requirements.txt
for a in "$@"; do case "$a" in
  --local) REQ=requirements-local.txt;; --server) REQ=requirements-server.txt;; --dev) REQ=requirements-dev.txt;;
  *) die "unknown flag $a";; esac; done
command -v python3 >/dev/null || die "python3 not found"
say "Creating venv at $VENV"
python3 -m venv "$VENV"
activate
pip install --upgrade pip wheel
pip install -r "$ROOT/$REQ"
mkdir -p "$ROOT/omni_data"
python -m omnipocket doctor || true
say "Done. UI: ./scripts/run.sh   Server: ./scripts/serve_gpu.sh"
