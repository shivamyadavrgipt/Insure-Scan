#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VENV_PYTHON="$PWD/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  python3 -m venv .venv
fi
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r backend/requirements.txt
exec "$VENV_PYTHON" demo_server.py
