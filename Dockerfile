# --- Base image ---
FROM python:3.11-slim

# 환경 변수(버퍼링 끄기 / .pyc 방지)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 작업 경로
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사 (server_quali.py, index.html, tools/, config.json 등)
COPY . .

# Cloud Run은 $PORT를 주입(기본 8080)
ENV PORT=8080

# 컨테이너 시작 커맨드
CMD ["uvicorn", "server_quali:app", "--host", "0.0.0.0", "--port", "8080"]
