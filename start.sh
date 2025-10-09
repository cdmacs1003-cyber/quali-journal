#!/bin/sh
set -eu
PORT="${PORT:-8080}"

if [ -f "admin/server_quali.py" ]; then
  TARGET="admin.server_quali:app"
elif [ -f "server_quali.py" ]; then
  TARGET="server_quali:app"
else
  echo "[FATAL] server_quali.py not found (looked in ./ and ./admin/)" >&2
  exit 1
fi

exec python -m uvicorn "$TARGET" --host 0.0.0.0 --port "$PORT" --workers 1
