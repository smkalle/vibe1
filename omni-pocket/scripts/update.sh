#!/usr/bin/env bash
# Pull the latest code and refresh dependencies in the existing venv.
source "$(dirname "$0")/common.sh"
cd "$ROOT"
git pull --ff-only
activate
REQ=requirements.txt
python -c 'import torch' 2>/dev/null && REQ=requirements-local.txt
python -c 'import fastapi, torch' 2>/dev/null && REQ=requirements-server.txt
pip install --upgrade -r "$REQ"
say "Updated ($REQ)"
