# ===== Base =====
FROM python:3.11-slim

# Python 런타임 기본 설정(버퍼링X, 바이트코드X, UTF-8)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# (옵션) 런타임에만 필요한 최소 패키지 설치
# - psycopg2-binary는 추가 시스템 패키지 없이도 동작
# - weasyprint 같은 무거운 도구는 optional에서 선택적으로 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ===== App files =====
# 작업 디렉터리 생성/이동
WORKDIR /app

# 의존성 파일만 먼저 복사 → 캐시 최적화
# (둘 다 있으면 optional도 설치하도록 조건 처리)
COPY requirements.txt ./requirements.txt
COPY requirements-optional.txt ./requirements-optional.txt

# ===== Dependencies =====
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements.txt \
 && if [ -f "requirements-optional.txt" ]; then \
        python -m pip install -r requirements-optional.txt; \
    fi

# 애플리케이션 전체 복사
# (dockerignore/gcloudignore로 불필요 파일 제외, HTML은 !규칙으로 포함)
COPY . /app

# ===== Runtime =====
# Cloud Run은 $PORT를 주입함 → 없으면 8080 기본값 사용
ENV PORT=8080

# (선택) 컨테이너 정보 노출 최소화
EXPOSE 8080

# ===== Start =====
# server_quali.py가 admin 폴더에 있으므로 모듈 경로는 admin.server_quali:app
CMD ["sh", "-c", "uvicorn admin.server_quali:app --host 0.0.0.0 --port ${PORT:-8080}"]
