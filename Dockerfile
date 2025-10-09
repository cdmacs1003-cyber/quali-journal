FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# ★ 여기서 권한 부여(윈도우에서도 OK)
RUN chmod +x /app/start.sh
EXPOSE 8080
CMD ["sh","-c","exec /app/start.sh"]


