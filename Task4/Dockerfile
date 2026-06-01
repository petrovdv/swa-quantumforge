FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (нужно для PII scanner)
RUN python -m spacy download en_core_web_lg

COPY rag_service .

RUN mkdir -p /app/chroma_db
RUN mkdir -p /app/models_cache

ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]