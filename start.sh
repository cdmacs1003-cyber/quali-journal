#!/bin/sh
set -eu

PORT="${PORT:-8080}"

# 자동 경로 감지: admin/server_quali.py 우선, 없으면 루트 server_quali.py
TARGET="server_quali:app"
if [ -f "admin/server_quali.py" ]; then
  TARGET="admin.server_quali:app"
elif [ ! -f "server_quali.py" ] && [ -f "/app/admin/server_quali.py" ]; then
  TARGET="admin.server_quali:app"
fi

exec python -m uvicorn "$TARGET" --host 0.0.0.0 --port "$PORT" --workers 1

