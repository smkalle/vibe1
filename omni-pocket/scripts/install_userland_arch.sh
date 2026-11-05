#!/usr/bin/env bash
# Omni Pocket installer for Arch Linux ARM inside UserLAnd or Termux proot-distro.
#   ./scripts/install_userland_arch.sh            # UI + mock/remote engines
#   ./scripts/install_userland_arch.sh --local    # + PyTorch/Transformers baseline engine
#   ./scripts/install_userland_arch.sh --server   # + FastAPI server (serve to other devices)
source "$(dirname "$0")/common.sh"
WITH=core
for a in "$@"; do case "$a" in --local) WITH=local;; --server) WITH=server;; *) die "unknown flag $a";; esac; done

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
command -v pacman >/dev/null || die "pacman not found - this script targets Arch Linux ARM"

say "Updating system + installing base tools"
$SUDO pacman -Syu --noconfirm --needed python python-pip git base-devel ffmpeg libsndfile curl

PY=python
if [[ "$WITH" != core ]] && ! python -c 'import sys; sys.exit(sys.version_info >= (3, 14))'; then
  warn "system Python may not have torch wheels yet; using uv to fetch Python 3.12"
  $SUDO pacman -S --noconfirm --needed uv || pip install --user --break-system-packages uv
  uv python install 3.12
  PY="$(uv python find 3.12)"
fi

say "Creating venv at $VENV"
"$PY" -m venv "$VENV"
activate
pip install --upgrade pip wheel
case "$WITH" in
  core)   pip install -r "$ROOT/requirements.txt" ;;
  local)  pip install --extra-index-url https://download.pytorch.org/whl/cpu -r "$ROOT/requirements-local.txt" ;;
  server) pip install --extra-index-url https://download.pytorch.org/whl/cpu -r "$ROOT/requirements-server.txt" ;;
esac

FREE="$(free_gb)"
if [[ "$WITH" != core && "$FREE" -lt 10 ]]; then
  warn "only ${FREE} GB free; the model needs ~5 GB in ~/.cache/huggingface plus headroom"
fi
mkdir -p "$ROOT/omni_data"
python -m omnipocket doctor || true
say "Done. Start with:  ./scripts/run.sh   and open http://localhost:8501 in Chrome on the phone"
