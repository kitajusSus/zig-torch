#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --upgrade uv
fi

uv venv .venv
. .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
