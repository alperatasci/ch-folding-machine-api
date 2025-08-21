# ---------- Build a tiny FastAPI image ----------
    FROM python:3.11-slim

    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # system deps for psycopg2 + certs
    RUN apt-get update && apt-get install -y --no-install-recommends \
          build-essential libpq-dev ca-certificates curl \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    COPY requirements.txt /app/
    RUN pip install --no-cache-dir -r requirements.txt
    
    COPY app/ /app/
    
    EXPOSE 8080
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
    