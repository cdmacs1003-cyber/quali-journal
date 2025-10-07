FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

# 앱 루트
WORKDIR /app

# 라이브러리 먼저 설치 (캐시 효율)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 소스 전체 복사 (admin, tools, data, etc.)
COPY . /app/

# Cloud Run 기본 포트
EXPOSE 8080

# FastAPI 실행 위치를 admin으로 맞춤
WORKDIR /app/admin
CMD ["uvicorn", "server_quali:app", "--host", "0.0.0.0", "--port", "8080"]
