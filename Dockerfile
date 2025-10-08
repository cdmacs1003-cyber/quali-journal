# ---- 베이스 이미지 ----
FROM python:3.11-slim

# 환경 설정: 버퍼링 비활성화, pyc 파일 생성 금지, UTF-8, 캐시 미사용
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# 코드가 위치할 루트 폴더 설정
WORKDIR /app

# 소스 전체 복사
COPY . /app

# requirements.txt가 있으면 설치, 없으면 FastAPI/Uvicorn/Pydantic 등 최소 패키지 설치
RUN set -e; \
    if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir fastapi uvicorn[standard] pydantic==1.* python-multipart; \
    fi

# Cloud Run 기본 포트
EXPOSE 8080

# 컨테이너 실행 시 FastAPI 앱 시작
# Cloud Run이 전달하는 PORT 환경변수를 그대로 사용(없으면 8080)
CMD ["sh","-c","python -m uvicorn server_quali:app --host 0.0.0.0 --port ${PORT:-8080}"]
