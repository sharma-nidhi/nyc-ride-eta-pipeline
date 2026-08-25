# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import pandas as pd

from fastapi import FastAPI, HTTPException

from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
)
from src.serving.model_loader import load_model, get_champion_info


# ---------------------------------------------------------------------------
# Eager model loading (works for both uvicorn and TestClient)
# ---------------------------------------------------------------------------

model, feature_pipeline = load_model()
champion_info = get_champion_info()


app = FastAPI(
    title="NYC Ride ETA Predictor",
    description="Production-grade ML pipeline for predicting trip durations.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    """Simple uptime check. Returns 200 if model is loaded."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfoResponse, summary="Model metadata")
async def model_info():
    """Return champion model metadata (type, metrics, run ID)."""
    return ModelInfoResponse(
        model_type=champion_info["run_name"],
        run_id=champion_info["run_id"],
        metrics=champion_info["metrics"],
        feature_count=champion_info["feature_count"],
        schema_version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, summary="Single prediction")
async def predict(req: PredictionRequest):
    """Predict ETA for a single trip request."""
    try:
        df = pd.DataFrame([req.model_dump()])
        features = feature_pipeline.transform(df)
        prediction = model.predict(features)
        return PredictionResponse(eta_seconds=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch prediction")
async def predict_batch(batch_req: BatchPredictionRequest):
    """Predict ETA for multiple trip requests (max 100 per batch)."""
    try:
        records = [req.model_dump() for req in batch_req.requests]
        df = pd.DataFrame(records)
        features = feature_pipeline.transform(df)
        predictions = model.predict(features)
        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(eta_seconds=float(p)) for p in predictions
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
