FROM python:3.11-slim

# 표준 출력 버퍼링과 pyc 파일 생성을 비활성화하고, UTF-8 인코딩과 PIP 캐시 비활성화를 설정합니다.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# 애플리케이션 루트 디렉터리를 설정합니다.
WORKDIR /app

# 전체 소스를 컨테이너에 복사합니다.
# 소스 디렉터리에는 server_quali.py와 기타 코드가 포함되어 있어야 합니다.
COPY . /app

# 의존성 설치:
#  - requirements.txt 파일이 있을 경우 그 내용을 설치하고,
#  - 없다면 FastAPI와 Uvicorn, Pydantic v1, python‑multipart 패키지를 기본으로 설치합니다.
RUN set -e; \
    if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir fastapi uvicorn[standard] pydantic==1.* python-multipart; \
    fi

# Cloud Run은 기본적으로 8080 포트를 사용하므로, 해당 포트를 노출합니다.
EXPOSE 8080

# 컨테이너 실행 시, Cloud Run이 제공하는 PORT 환경변수를 읽어 해당 포트로 서버를 기동합니다.
# 만약 PORT 값이 없다면 8080을 사용합니다. server_quali.py가 루트에 있다고 가정합니다.
CMD ["sh", "-c", "python -m uvicorn server_quali:app --host 0.0.0.0 --port ${PORT:-8080}"]