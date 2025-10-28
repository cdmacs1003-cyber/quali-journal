FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 소스 전체 복사(정적/템플릿 포함)
COPY . /app

# Cloud Run 표준 포트
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PORT=8080

# uvicorn 0.0.0.0:8080 고정
CMD ["uvicorn","server_quali:app","--host","0.0.0.0","--port","8080"]
