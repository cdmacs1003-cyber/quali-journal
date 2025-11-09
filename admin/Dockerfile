# QualiJournal Admin — Dockerfile (Kaniko-safe · dist 해시 빌드 · 8080 고정)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

# 최소 OS 패키지
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# 1) pip 업그레이드 + requirements(있으면) 설치 + 필수 런타임 보강(누락만 설치)
RUN set -eux; \
    python -m pip install --no-warn-script-location --upgrade pip; \
    for f in /app/requirements.txt /app/requirements-prod.txt /app/admin/requirements.txt /app/admin/requirements-prod.txt; do \
      if [ -f "$f" ]; then python -m pip install -r "$f"; fi; \
    done; \
    python -c "import importlib.util as u, subprocess, sys; \
pkgs='uvicorn fastapi starlette python-multipart jinja2 markdown aiofiles python-dotenv'.split(); \
miss=[p for p in pkgs if u.find_spec(p) is None]; \
print('missing:',miss); \
sys.exit(0) if not miss else subprocess.check_call([sys.executable,'-m','pip','install','--no-warn-script-location',*miss])"

# 2) UI 해시 빌드(dist 생성) — tools 경로 자동 탐지 (없으면 skip)
RUN set -eux; \
    if command -v python >/dev/null 2>&1; then \
      if   [ -f /app/tools/build_assets_hash.py ]; then python /app/tools/build_assets_hash.py --clean; \
      elif [ -f /app/admin/tools/build_assets_hash.py ]; then python /app/admin/tools/build_assets_hash.py --clean; \
      elif [ -f /app/build_assets_hash.py ]; then python /app/build_assets_hash.py --clean; \
      else echo 'skip: build_assets_hash.py not found in context'; fi; \
    else echo 'skip: python not found'; fi

# 비루트 사용자(선택)
RUN useradd -m -u 10001 appuser || true
USER appuser

# /app/admin이 있으면 그 디렉터리에서 실행(컨텍스트에 무관하게 일관성)
WORKDIR /app/admin
EXPOSE 8080

# $PORT 준수(Cloud Run)
CMD ["sh","-c","python -m uvicorn server_quali:app --host 0.0.0.0 --port ${PORT:-8080}"]
