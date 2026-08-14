# Packaging for the inference service (Week 3 / M4).
# Placeholder — finalize once serving/api.py is implemented.
FROM python:3.11-slim

WORKDIR /app

# Install only what serving needs (keep the image small)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + model artifacts
COPY serving/ serving/
COPY features/ features/
COPY models/ models/
COPY config/ config/

EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
