FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

# 앱 루트
WORKDIR /app

## -----------------------------------------------------------------------------
## Install dependencies and copy source
##
## Our repository may or may not include a requirements.txt file. Attempting to
## copy a missing file causes a build failure (the default behaviour of the
## Docker `COPY` instruction). To make the build robust, we first copy all
## sources into the container, then conditionally install dependencies.
##
# Copy the entire application into the image. This includes all Python
# modules, HTML files and configuration. The context is the repository root.
COPY . /app

# Install Python dependencies. If a requirements.txt file exists in the
# repository root it will be used; otherwise a minimal set of packages
# required to run the FastAPI application is installed. We use a single RUN
# statement with a shell to ensure conditional logic works correctly.
RUN set -e; \
    if [ -f requirements.txt ]; then \
      echo "Installing dependencies from requirements.txt"; \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      echo "requirements.txt not found; installing minimal dependencies"; \
      pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-multipart; \
    fi

## -----------------------------------------------------------------------------
## Configure runtime
##
# Expose Cloud Run's default port. Cloud Run will communicate with the
# container over this port.
EXPOSE 8080

# When running on Cloud Run we want to execute the FastAPI application from
# the `admin` subdirectory. The server code defines the FastAPI app in
# `server_quali.py`. Set the working directory accordingly.
WORKDIR /app/admin

# Start the FastAPI application using uvicorn. The host must be 0.0.0.0 so
# Cloud Run can connect to the container, and the port must match the one
# exposed above (8080).
CMD ["uvicorn", "server_quali:app", "--host", "0.0.0.0", "--port", "8080"]
