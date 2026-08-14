"""
FastAPI inference service for ride ETA (Week 3 / M4).

PLACEHOLDER — implement endpoints. Load the model + schema ONCE at startup and reuse
`features.build_features` so serving matches training exactly.

Endpoints:
  GET  /health   -> {"status": "ok", "model_version": ...}
  POST /predict  -> TripRequest in, ETAResponse out; logs every prediction.

Run:  uvicorn serving.api:app --reload --port 8000   (docs at /docs)
"""
from fastapi import FastAPI

from serving.schemas import TripRequest, ETAResponse

app = FastAPI(title="NYC Ride ETA API", version="0.1")

# TODO Week 3: load model + feature_schema.json at startup
# TODO Week 4: from monitoring.logger import init, log ; init()

MODEL_VERSION = "eta-v1"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=ETAResponse)
def predict(trip: TripRequest):
    raise NotImplementedError("TODO Week 3: build features -> model.predict -> ETA")
