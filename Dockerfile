FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model at build time (for this demo)
RUN python -m src.ml_service.model

EXPOSE 8000

CMD ["uvicorn", "src.ml_service.server:app", "--host", "0.0.0.0", "--port", "8000"]
