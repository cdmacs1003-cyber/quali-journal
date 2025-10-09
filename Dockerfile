FROM python:3.11-slim

# avoid .pyc, make stdout/err unbuffered, and enforce utf-8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# 1) install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) copy the rest of the source
COPY . .

# 3) make start script executable (no-op on Windows hosts)
RUN chmod +x /app/start.sh || true

EXPOSE 8080

# Cloud Run will pass PORT; we delegate to start.sh (detects module path & runs uvicorn)
CMD ["sh","-c","exec /app/start.sh"]
