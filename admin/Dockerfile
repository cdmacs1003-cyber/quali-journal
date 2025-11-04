# =========================================================
# QualiJournal Admin — Dockerfile (SSOT: admin/ 컨텍스트 고정)
# - Cloud Run 이미지 배포용
# - 엔트리: server_quali:app   (접두사 없음)
# =========================================================
FROM python:3.11-slim

# 기본 런타임 환경
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# (옵션) 일부 패키지 빌드를 위한 최소 빌드 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 앱 작업 디렉터리
WORKDIR /app

# 종속성 캐시 최적화: 먼저 requirements만 복사
COPY requirements.txt /app/requirements.txt

# 파이썬 의존성 설치 (필요시 프레임워크 기본 패키지도 함께 보강)
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install "fastapi" "uvicorn[standard]" gunicorn python-dotenv

# 애플리케이션 복사 (admin/ 컨텍스트의 모든 파일)
COPY . /app

# 최신 admin/index.html을 최종 산출물로 강제 + 오래된 dist 제거 (멱등)
RUN set -eux; \
    if [ -f "/app/admin/index.html" ]; then \
        cp -f /app/admin/index.html /app/index.html; \
    fi; \
    rm -rf /app/dist

# (선택) 빌드 도구 제거로 경량화
RUN apt-get purge -y build-essential || true && apt-get autoremove -y || true

# 보안상 비루트 사용자로 실행
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run 포트
EXPOSE 8080

# ✅ ASGI 엔트리 고정: server_quali:app (admin 컨텍스트이므로 접두사 없음)
CMD ["bash","-lc","exec gunicorn -k uvicorn.workers.UvicornWorker -b :$PORT server_quali:app"]
