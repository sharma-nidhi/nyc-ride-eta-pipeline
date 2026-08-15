"""
FastAPI inference service for NYC ride ETA.

Loads the trained model once at startup and reuses features.build_features so the
features here are IDENTICAL to training (no training-serving skew).

Run:  uvicorn serving.api:app --reload --port 8000     (interactive docs at /docs)
"""
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from features.build_features import build_features
from serving.schemas import TripRequest, ETAResponse

MODEL_PATH = "models/eta-v1.joblib"
SCHEMA_PATH = "models/feature_schema.json"
MODEL_VERSION = "eta-v1"

app = FastAPI(title="NYC Ride ETA API", version="1.0")

# Load once at startup (not on every request)
model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f:
    FEATURE_COLS = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=ETAResponse)
def predict(trip: TripRequest):
    # 1. One-row DataFrame with the raw column names build_features expects
    row = pd.DataFrame([trip.model_dump()])

    # 2. Same feature logic as training
    X = build_features(row)[FEATURE_COLS]

    # 3. Model predicts log1p(seconds); invert with expm1
    eta_seconds = float(np.expm1(model.predict(X)[0]))

    return ETAResponse(
        eta_seconds=round(eta_seconds, 1),
        eta_minutes=round(eta_seconds / 60, 1),
        model_version=MODEL_VERSION,
    )