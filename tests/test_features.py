"""
Unit tests for the feature pipeline (Week 1 logic, verified). Run:  pytest -q
"""
import json
import pandas as pd
from features.build_features import build_features, haversine_km


def sample_df():
    # 2016-03-14 is a Monday at 17:xx — a known reference row
    return pd.DataFrame([{
        "pickup_datetime": "2016-03-14 17:24:55",
        "pickup_latitude": 40.7484, "pickup_longitude": -73.9857,
        "dropoff_latitude": 40.7580, "dropoff_longitude": -73.9850,
        "passenger_count": 1, "vendor_id": 2, "store_and_fwd_flag": "N",
    }])


def test_haversine_known_distance():
    # 1 degree of latitude is ~111 km
    assert 110 < haversine_km(0.0, 0.0, 1.0, 0.0) < 112


def test_feature_columns_match_schema():
    with open("models/feature_schema.json") as f:
        schema = json.load(f)
    assert list(build_features(sample_df()).columns) == schema


def test_no_target_leakage():
    # serving never knows the answer, so features must not include the target
    assert "trip_duration" not in build_features(sample_df()).columns


def test_temporal_features():
    X = build_features(sample_df())
    assert X.loc[0, "hour"] == 17
    assert X.loc[0, "day_of_week"] == 0     # Monday
    assert X.loc[0, "is_weekend"] == 0
    assert X.loc[0, "month"] == 3


def test_distances():
    X = build_features(sample_df())
    assert X.loc[0, "haversine_km"] > 0
    # grid distance is never shorter than straight-line distance
    assert X.loc[0, "manhattan_km"] >= X.loc[0, "haversine_km"]