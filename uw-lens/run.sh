#!/usr/bin/env bash
# Quick start: venv, deps, sample corpus, index, UI.  Usage: ./run.sh [--gemma]
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
if [[ "${1:-}" == "--gemma" ]]; then
  pip install -q -r requirements-gemma.txt
else
  pip install -q -r requirements.txt
  export UWLENS_EMBEDDER="${UWLENS_EMBEDDER:-hashing}"
fi
[[ -d sample_corpus ]] || python -m uwlens sample
python -m uwlens index
exec streamlit run app.py
