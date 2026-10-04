#!/usr/bin/env bash
# One-command setup and launch for the jev-call-lab workbench (Linux/macOS, incl. Arch/Omarchy).
#
#   ./run.sh          create .venv if needed, install deps, start the workbench (http://localhost:8501)
#   ./run.sh check    same setup, then run the offline test suite and evals E1-E7 instead
#
# Arch-based systems mark the system Python as externally managed, so everything goes into .venv.
# The OpenRouter key is entered in the app (session memory only); nothing here reads or stores it.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || { echo "Python 3.10+ is required (found $("$PY" --version 2>&1))." >&2; exit 1; }

if [ ! -x .venv/bin/python ]; then
    echo "Creating .venv ..."
    "$PY" -m venv .venv
fi
if [ ! -f .venv/.deps-ok ] || [ requirements.txt -nt .venv/.deps-ok ]; then
    echo "Installing dependencies ..."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
    touch .venv/.deps-ok
fi

case "${1:-app}" in
    check)
        .venv/bin/python -m pytest -q tests
        .venv/bin/python run_evals.py
        ;;
    app)
        exec .venv/bin/python -m streamlit run app.py
        ;;
    *)
        echo "usage: $0 [app|check]" >&2
        exit 2
        ;;
esac
