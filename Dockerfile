# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# deps 설치
COPY admin/requirements.txt /app/admin/requirements.txt
RUN pip install --no-cache-dir -r /app/admin/requirements.txt

# 앱 소스(루트 기준으로 묶기)
COPY admin/ /app/admin/
COPY tools/ /app/tools/
COPY feeds/ /app/feeds/
COPY config.json /app/config.json
# (옵션) 오케스트레이터 사용 시
COPY orchestrator.py /app/orchestrator.py

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080

CMD ["sh","-c","uvicorn admin.server_quali:app --host 0.0.0.0 --port ${PORT:-8080}"]