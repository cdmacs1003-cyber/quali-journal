FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

# Set working directory inside the container
WORKDIR /app

# Copy dependency file if present and install Python dependencies.
# Cloud Run's build environment may not include your own requirements.txt. To ensure the
# application starts successfully, we install from requirements.txt when it exists,
# otherwise fall back to installing the minimal set of packages needed by server_quali.py.
COPY requirements.txt /app/ 2>/dev/null || true
RUN set -e; \
    if [ -f "requirements.txt" ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-multipart; \
    fi

# Copy source code into the container
COPY . /app/

EXPOSE 8080

# Run the FastAPI application via uvicorn. We run server_quali:app directly since
# server_quali.py defines the FastAPI application and root route. The host and port
# are set to values Cloud Run expects.
CMD ["uvicorn", "server_quali:app", "--host", "0.0.0.0", "--port", "8080"]
