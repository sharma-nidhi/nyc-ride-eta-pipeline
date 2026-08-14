"""
Feature engineering — the SINGLE source of truth for transforming raw trips into
model features (Week 1 / M2).

CRITICAL: training (training/train.py) and serving (serving/api.py) MUST both call
`build_features()` here. Duplicating this logic anywhere causes training-serving skew.

PLACEHOLDER — implement the transformations below and persist a feature store +
feature_schema.json (the contract between training and serving).

Planned features:
  - Temporal: hour-of-day, day-of-week, weekend flag, month/season (from pickup_datetime).
  - Distance: haversine(pickup_lat/lon, dropoff_lat/lon); optionally Manhattan distance.
  - Location: pickup/dropoff zone or borough bucket (optional).
  - Weather: join external weather by date/hour (optional enrichment).
  - Target: trip_duration -> log1p (predict in log space, invert at serving).
"""
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw trips -> ML-ready feature frame. Used by BOTH train and serve."""
    raise NotImplementedError("TODO Week 1: implement shared feature logic")


if __name__ == "__main__":
    # TODO: load validated data, build features, write feature store + schema json
    print("build_features placeholder")
