# ---- base runtime ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS deps (필요시 보강 가능)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- install deps ----
# requirements.txt가 있으면 먼저 복사/설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt || true

# ---- copy app code (핵심: orchestrator/ admin/ tools/ 포함) ----
COPY orchestrator.py /app/orchestrator.py
COPY admin/ /app/admin/
COPY tools/ /app/tools/
# 선택: 설정 파일/피드가 있다면 함께
COPY config.json /app/config.json
# 선택: 정적 seed 데이터(없어도 런타임에 생성됨)
# COPY data/ /app/data/

# 런타임용 작업/아카이브 디렉터리 확보
RUN mkdir -p /app/archive /app/logs

# Cloud Run은 $PORT를 주입; uvicorn은 server_quali.py 안에서 실행
# (server_quali.py __main__ 에서 uvicorn.run 호출) :contentReference[oaicite:6]{index=6}
EXPOSE 8080
ENV PORT=8080

# ---- run ----
# server_quali.py는 admin 폴더에 있음
CMD ["python", "-u", "admin/server_quali.py"]
