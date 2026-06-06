FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY docker/requirements-model.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/opensky ./models/opensky

EXPOSE 8001

CMD ["sh", "-c", "uvicorn src.model_service.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
