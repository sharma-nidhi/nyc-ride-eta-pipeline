# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import joblib
import pathlib
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, FunctionTransformer

logger = logging.getLogger(__name__)

PIPELINE_PATH = pathlib.Path("models/feature_pipeline.pkl")

# Columns consumed by each transform — matches feature_meta.json at_event entries only
NUMERIC_COLS = ["passenger_count", "hour_sin", "hour_cos", "day_of_week", "haversine_dist_km"]
CATEGORICAL_COLS = ["vendor_id", "store_and_fwd_flag"]
PASSTHROUGH_COLS = ["is_weekend", "is_rush_hour"]


def _extract_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Derives cyclical and binary time features from pickup_datetime."""
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"])
    hour = dt.dt.hour

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("int8")

    # Rush hour: 7-9 AM and 4-7 PM
    df["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype("int8")

    return df.drop(columns=["pickup_datetime"])


def _compute_haversine(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces four coordinate columns with a single great-circle distance in km."""
    df = df.copy()
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine_dist_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    return df.drop(columns=["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])


def build_pipeline() -> Pipeline:
    """Constructs the full feature pipeline. Shared by training and serving."""
    custom_transforms = Pipeline([
        ("temporal", FunctionTransformer(_extract_temporal)),
        ("spatial",  FunctionTransformer(_compute_haversine)),
    ])

    col_transform = ColumnTransformer([
        ("num", RobustScaler(), NUMERIC_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
    ], remainder="passthrough")  # passthrough keeps is_weekend, is_rush_hour unchanged

    return Pipeline([
        ("custom_features",     custom_transforms),
        ("standard_transforms", col_transform),
    ])


def save_pipeline(pipeline: Pipeline) -> None:
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info("Pipeline saved to %s", PIPELINE_PATH)


def load_pipeline() -> Pipeline:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {PIPELINE_PATH}. Run preprocess.py first.")
    return joblib.load(PIPELINE_PATH)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.path.append(".")

    from src.data.ingest import load_raw
    from src.data.validate import validate_data

    df_raw = load_raw(sample_mode=True, sample_size=10_000)
    df_clean, _ = validate_data(df_raw)

    pipeline = build_pipeline()
    X = df_clean.drop(columns=["trip_duration"])
    X_out = pipeline.fit_transform(X)

    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {X_out.shape}")
    print("Feature pipeline test passed.")
    save_pipeline(pipeline)
    print(f"Pipeline saved to {PIPELINE_PATH}")