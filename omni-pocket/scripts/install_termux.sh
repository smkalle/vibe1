#!/usr/bin/env bash
# Omni Pocket installer for plain Termux (F-Droid / GitHub build of Termux).
# Installs the UI + mock/remote engines. The model itself runs either in
# Chrome (WebGPU tab) or on a GPU box (remote engine) - PyTorch has no
# Android wheels, so the 'transformers' engine needs UserLAnd/proot instead.
source "$(dirname "$0")/common.sh"
[[ "$(detect_env)" == termux ]] || die "this script is for Termux; see guides/ for other environments"

say "Updating packages"
pkg update -y && pkg upgrade -y

say "Installing prebuilt Python stack (avoids compiling numpy/pyarrow on the phone)"
pkg install -y python python-pip git python-numpy python-pillow python-pyarrow \
               rust binutils clang make pkg-config libjpeg-turbo termux-tools

say "pandas (Streamlit dependency)"
if ! python -c 'import pandas' 2>/dev/null; then
  pkg install -y python-pandas 2>/dev/null \
    || { pkg install -y tur-repo && pkg update -y && pkg install -y python-pandas; } 2>/dev/null \
    || warn "no prebuilt pandas package - pip will build it (20-40 min)"
fi

# Rust-based wheels (rpds-py for jsonschema/altair) build via maturin and need this:
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk 2>/dev/null || echo 24)"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"   # keep the phone from overheating

say "Creating venv (with system site-packages so it reuses numpy/pyarrow/pandas)"
python -m venv --system-site-packages "$VENV"
activate
pip install --upgrade pip wheel
say "Installing Streamlit + client deps (first time can take 10-20 min: rpds-py/tornado compile)"
pip install -r "$ROOT/requirements.txt"

if ! command -v termux-tts-speak >/dev/null 2>&1; then
  warn "Optional: 'pkg install termux-api' + the Termux:API app adds termux-tts-speak voice output"
fi
mkdir -p "$ROOT/omni_data"
python -m omnipocket doctor || true
say "Done. Start with:  ./scripts/run.sh   (then Chrome opens http://localhost:8501)"
