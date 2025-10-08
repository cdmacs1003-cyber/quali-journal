FROM python:3.11-slim

# 운영 환경에 맞춰 표준 출력 버퍼링과 pyc 파일 생성을 비활성화하고,
# UTF-8 인코딩과 pip 캐시 비활성화를 설정합니다.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# 애플리케이션 루트 경로
WORKDIR /app

# 소스 전체를 복사합니다. server_quali.py는 레포 루트에 있어야 합니다.
COPY . /app

# requirements.txt가 있으면 이를 사용하여 패키지를 설치하고,
# 없으면 FastAPI/Uvicorn/Pydantic/pymultipart 최소 패키지를 설치합니다.
RUN set -e; \
    if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir fastapi uvicorn[standard] pydantic==1.* python-multipart; \
    fi

# Cloud Run 기본 포트
EXPOSE 8080

# 서버 실행: Cloud Run에서 넘겨주는 PORT 환경변수를 사용하여 uvicorn을 실행합니다.
# PORT 값이 없을 경우 8080을 사용합니다.
CMD ["sh", "-c", "python -m uvicorn server_quali:app --host 0.0.0.0 --port ${PORT:-8080}"]