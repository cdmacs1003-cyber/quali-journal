FROM python:3.11-slim

# Ensure Python prints to stdout/stderr immediately and doesn't write .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# Set the working directory in the container. All application code will live under /app
WORKDIR /app

# Copy the entire application source code into the container. This includes
# server_quali.py, main.py, templates, static files and any auxiliary modules.
COPY . /app

# Install Python dependencies. If a requirements.txt is present in the repo, use it; otherwise
# install the minimal set of packages required to run the QualiJournal admin API. This fallback
# avoids build failures when requirements.txt is absent but ensures FastAPI and Uvicorn are available.
RUN set -e; \
    if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-multipart; \
    fi

# Cloud Run expects the application to listen on port 8080 by default. Expose it here for clarity.
EXPOSE 8080

# Start the FastAPI application using uvicorn. Use the module and app name directly; the
# host must be 0.0.0.0 so that it is accessible from outside the container.
# The port is set to 8080, which matches Cloud Run's default.
CMD ["uvicorn", "server_quali:app", "--host", "0.0.0.0", "--port", "8080"]