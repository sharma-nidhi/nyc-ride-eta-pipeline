# Packaging for the inference service (Week 3 / M4).
FROM python:3.11-slim

WORKDIR /app

# Install only what serving needs (slim deps -> smaller, faster image)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + model artifacts
COPY serving/ serving/
COPY features/ features/
COPY models/ models/
COPY config/ config/
COPY monitoring/ monitoring/

EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
