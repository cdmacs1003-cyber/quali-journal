# ===========================
# QualiJournal Admin Dockerfile
# ===========================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# (선택) 빌드 툴킷 설치 후 제거 — 일부 패키지가 wheel이 없을 때 대비
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 종속성 레이어 캐시 최적화
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install "fastapi" "uvicorn[standard]" "gunicorn" "python-dotenv"

# 애플리케이션 복사
COPY . /app

# (옵션) 빌드툴 제거로 경량화
RUN apt-get purge -y build-essential || true && apt-get autoremove -y || true

# 권장: 비루트 사용자
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run 표준: $PORT 사용 + ASGI 엔트리 = server_quali:app
CMD ["bash","-lc","exec gunicorn -k uvicorn.workers.UvicornWorker -b :$PORT server_quali:app"]
