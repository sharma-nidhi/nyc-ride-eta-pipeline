"""
Lightweight prediction logger for monitoring.
Appends incoming requests and model predictions to a JSONL file for downstream drift analysis.
"""
import json
import time
import pandas as pd
import pathlib
import logging

logger = logging.getLogger(__name__)

LOG_PATH = pathlib.Path("data/monitoring/production_logs.jsonl")


def _canonicalize_pickup_datetime(row: dict) -> dict:
    """Normalize pickup_datetime to UTC ISO-8601 (YYYY-MM-DDTHH:MM:SSZ)."""
    if "pickup_datetime" not in row:
        return row
    try:
        ts = pd.to_datetime(row["pickup_datetime"], errors="coerce", utc=True)
        if pd.isna(ts):
            return row
        row["pickup_datetime"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # Keep logging resilient; validation happens upstream.
        return row
    return row


def log_prediction(features: pd.DataFrame, predictions, latency_ms: float):
    """
    Log a batch of predictions to the monitoring log.

    Parameters
    ----------
    features : pd.DataFrame
        The raw input features used for prediction (before pipeline transform).
    predictions : np.ndarray or list
        The model's predicted ETA seconds.
    latency_ms : float
        Inference time in milliseconds.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Handle numpy arrays (feature names are lost after sklearn pipeline transform)
    if isinstance(features, pd.DataFrame):
        rows = features.to_dict(orient="records")
    else:
        # numpy array: just log shape + prediction
        rows = [{"num_features": features.shape[1]}] * len(features)

    for row, pred in zip(rows, predictions):
        row = _canonicalize_pickup_datetime(row)
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "latency_ms": latency_ms,
            "prediction": float(pred),
            **row,
        }
        # Write atomically per line
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    logger.info("Logged %d predictions to %s", len(predictions), LOG_PATH)


def get_production_logs(limit: int = None) -> pd.DataFrame:
    """Load historical predictions for drift analysis."""
    if not LOG_PATH.exists():
        return pd.DataFrame()

    with open(LOG_PATH, "r") as f:
        lines = f.readlines()

    if limit:
        lines = lines[-limit:]

    data = [json.loads(line) for line in lines]
    return pd.DataFrame(data)
