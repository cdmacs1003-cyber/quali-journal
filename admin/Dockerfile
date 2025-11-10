# Python 3.11 slim
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir uvicorn gunicorn

# 소스 복사 (도커/클라우드 빌드 ignore 규칙으로 불필요물 제거)
COPY . /app
RUN python -m compileall -q /app

EXPOSE 8080
# Cloud Run의 PORT 환경변수를 존중하여 리슨
CMD ["sh","-c","python -m uvicorn server_quali:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info"]

# (선택) 헬스체크 힌트
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1
