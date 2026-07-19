# FastAPI + Aestra static UI
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-pfl.txt pyproject.toml ./
COPY cyclebench ./cyclebench
COPY api ./api
COPY web ./web
COPY models ./models
COPY data/nhanes_harmonized ./data/nhanes_harmonized
COPY data/foundation ./data/foundation
COPY results ./results
COPY docs ./docs

# Torch is optional (large). Soft read works without it; add requirements-pfl.txt for full pFL.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e . \
 && pip install --no-cache-dir openai

EXPOSE 8000

# Cloud hosts inject PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
