FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
CMD ["uvicorn","server_quali:app","--host","0.0.0.0","--port","8080"]
