# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pathlib
import logging
import joblib
import json
import hashlib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from typing import Tuple, Dict, Any

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features.feature_pipeline import load_pipeline

logger = logging.getLogger(__name__)

# Paths
PROCESSED_X = pathlib.Path("data/processed/X_train.parquet")
PROCESSED_Y = pathlib.Path("data/processed/y_train.parquet")
MODEL_OUTPUT_DIR = pathlib.Path("models/artifacts")
FEATURE_REGISTRY_PATH = pathlib.Path("data/contracts/feature_registry.json")
VALIDATION_REPORT_PATH = pathlib.Path("data/validation_report.json")

# MLflow Configuration (Local SQLite Backend)
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }

def train_baseline_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a Baseline Ridge Regression model and tracks it with MLflow.
    This establishes the performance floor for the project.
    """
    # 1. Split data (80/20) - Chronological split to prevent future leakage
    # Data is already sorted by pickup_datetime in ingest.py
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 2. Model Definition
    # Ridge adds L2 regularization to prevent overfitting on highly correlated features
    model = Ridge(alpha=1.0) 

    # 3. MLflow Tracking
    mlflow.set_experiment("NYC-ETA-Prediction")
    mlflow.sklearn.autolog(log_models=False) # Keep autolog params/metrics, manual log_model handles model artifact
    
    with mlflow.start_run(run_name="Baseline_Ridge"):
        logger.info("Training Baseline Ridge Model...")
        
        # We can still log custom parameters if needed
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_metric("x_null_fraction", float(X.isna().mean().mean()))

        if FEATURE_REGISTRY_PATH.exists():
            with open(FEATURE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)

            feature_names = [col["name"] for col in schema.get("feature_columns", [])]
            feature_hash = hashlib.sha256(
                json.dumps(feature_names, sort_keys=True).encode("utf-8")
            ).hexdigest()

            mlflow.log_params({
                "feature_set": schema.get("feature_set", "unknown"),
                "schema_version": schema.get("schema_version", "unknown"),
                "feature_count": len(feature_names),
                "dvc_slice": schema.get("dvc_slice", "unknown"),
                "feature_list_sha256": feature_hash,
            })
            mlflow.log_artifact(str(FEATURE_REGISTRY_PATH), artifact_path="contracts")
        else:
            logger.warning("Feature registry not found at %s", FEATURE_REGISTRY_PATH)

        if VALIDATION_REPORT_PATH.exists():
            mlflow.log_artifact(str(VALIDATION_REPORT_PATH), artifact_path="validation")

        # Train
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        predictions = model.predict(X_val)
        metrics = evaluate_model(y_val, predictions)
        
        # Log custom metrics (autolog handles some, but we want our specific set)
        mlflow.log_metrics(metrics)
        logger.info(f"Baseline Metrics: {metrics}")

        # Infer and log model signature (Input/Output Schema)
        signature = infer_signature(X_val, predictions)
        mlflow.sklearn.log_model(model, name="baseline_ridge_model", signature=signature)

        # Save Model Artifact for local accessibility
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_OUTPUT_DIR / "baseline_ridge.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

    return model, metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        logger.info("Starting Baseline Model Training...")
        
        # Load processed data
        X = pd.read_parquet(PROCESSED_X)
        y = pd.read_parquet(PROCESSED_Y).squeeze() # Ensure y is a Series
        
        # Execute training
        model, metrics = train_baseline_model(X, y)
        
        print("\n--- Baseline Training Complete ---")
        print(f"MAE:  {metrics['mae']:.2f}s")
        print(f"RMSE: {metrics['rmse']:.2f}s")
        print(f"R2:   {metrics['r2']:.4f}")
        print(f"\nExperiment tracked in MLflow. Run 'mlflow ui' to see results.")
        
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
