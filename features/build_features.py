
import json
import os
import numpy as np
import pandas as pd

PROCESSED_PATH = "data/processed/train_clean.parquet"
FEATURE_STORE = "data/processed/features.parquet"
SCHEMA_PATH = "models/feature_schema.json"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    pickup = pd.to_datetime(df["pickup_datetime"])

    out["hour"] = pickup.dt.hour
    out["day_of_week"] = pickup.dt.dayofweek           
    out["is_weekend"] = (pickup.dt.dayofweek >= 5).astype(int)
    out["month"] = pickup.dt.month

    out["haversine_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    out["manhattan_km"] = (
        haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                     df["pickup_latitude"], df["dropoff_longitude"])
        + haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                       df["dropoff_latitude"], df["pickup_longitude"])
    )

    out["passenger_count"] = df["passenger_count"]
    out["vendor_id"] = df["vendor_id"]
    out["store_and_fwd"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    return out


def save_feature_store(df: pd.DataFrame) -> None:
    """Build features + attach the target, then save the feature store and schema."""
    X = build_features(df)
    X["trip_duration"] = df["trip_duration"].values      # target sits beside the features

    os.makedirs("data/processed", exist_ok=True)
    X.to_parquet(FEATURE_STORE, index=False)

    feature_cols = [c for c in X.columns if c != "trip_duration"]
    os.makedirs("models", exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Feature store -> {FEATURE_STORE}  ({len(X):,} rows, {len(feature_cols)} features)")
    print(f"Schema        -> {SCHEMA_PATH}")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_PATH)
    save_feature_store(df)
    print("\nFeature engineering complete — feature store ready for training.")