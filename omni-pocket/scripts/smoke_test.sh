#!/usr/bin/env bash
# Offline end-to-end check: unit tests, CLI with the mock engine, then a
# headless Streamlit boot + health probe. Safe to run on the phone.
source "$(dirname "$0")/common.sh"
activate
cd "$ROOT"
PORT="${OMNI_SMOKE_PORT:-8599}"
if python -c 'import pytest' 2>/dev/null; then say "pytest"; python -m pytest -q tests; else warn "pytest not installed - skipping unit tests"; fi
say "CLI (mock engine)"
python -m omnipocket ask --engine mock --text "smoke test"
say "Streamlit boot on :$PORT"
OMNI_ENGINE=mock streamlit run app.py --server.port "$PORT" --server.address 127.0.0.1 >/dev/null 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then say "UI healthy"; exit 0; fi
  sleep 1
done
die "Streamlit did not become healthy on :$PORT"
