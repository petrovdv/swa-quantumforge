FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/chroma_db
RUN mkdir -p /app/models_cache

ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache
ENV PYTHONPATH="${PYTHONPATH}:/app/rag_service"

EXPOSE 8000

CMD ["uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8000"]