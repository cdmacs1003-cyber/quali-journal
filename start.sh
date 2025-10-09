#!/usr/bin/env bash
set -euo pipefail

# honour Cloud Run PORT; default locally
PORT="${PORT:-8080}"

# detect target app (prefer admin/server_quali.py)
if [[ -f "admin/server_quali.py" ]]; then
  TARGET="admin.server_quali:app"
elif [[ -f "server_quali.py" ]]; then
  TARGET="server_quali:app"
else
  echo "[FATAL] server_quali.py not found (looked in ./ and ./admin/)" >&2
  exit 1
fi

# utf-8 for all subprocess output
export PYTHONIOENCODING=UTF-8

# run uvicorn with the correct interpreter
exec python -m uvicorn "$TARGET" --host 0.0.0.0 --port "${PORT}"
