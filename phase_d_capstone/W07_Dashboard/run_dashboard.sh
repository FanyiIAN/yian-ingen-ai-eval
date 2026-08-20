#!/usr/bin/env bash
set -euo pipefail

DASHBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$DASHBOARD_DIR/.venv"
cd "$DASHBOARD_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -r "$DASHBOARD_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m streamlit run "$DASHBOARD_DIR/app.py" "$@"
