FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (for building some scientific libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY MarketPulse/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

# Run seeding script before starting backend
CMD ["sh", "-c", "python MarketPulse/database/init_db.py && uvicorn MarketPulse.backend.main:app --host 0.0.0.0 --port 8000"]
