# Python 런타임(슬림)
FROM python:3.11-slim

# 기본 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# 작업 루트
WORKDIR /app

# (1) 루트 requirements.txt 사용
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# (2) 앱 리소스 복사: admin/ 및 필요한 루트 파일들
COPY admin/ /app/admin/
COPY config.json /app/config.json
COPY archive/ /app/archive/
COPY feeds/ /app/feeds/
COPY data/ /app/data/
# 선택 JSON들(있으면 복사)
COPY selected_articles.json /app/selected_articles.json
COPY selected_keyword_articles.json /app/selected_keyword_articles.json

# Cloud Run 기본 포트
ENV PORT=8080
EXPOSE 8080

# (3) 앱 디렉터리로 이동
WORKDIR /app

# (4) Uvicorn으로 FastAPI 기동
# server_quali.py에 FastAPI app 존재 (/health 구현)
CMD ["bash","-lc","gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT admin.server_quali:app"]
