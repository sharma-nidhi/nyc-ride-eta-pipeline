# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Data & Feature Contract — Single Source of Truth
=================================================
This module defines the validation bounds and feature lists used across
the entire pipeline. Every layer imports from here to prevent train-serving
skew. Change a bound once → validation, training, and serving all update.
"""

# ---------------------------------------------------------------------------
# Raw Data Validation Bounds (shared by validate.py & schemas.py)
# ---------------------------------------------------------------------------

# NYC bounding box (used by validate.py and API input validation)
NYC_LAT_MIN = 40.5
NYC_LAT_MAX = 42.0
NYC_LON_MIN = -75.0
NYC_LON_MAX = -72.0

# Trip duration (seconds) — enforces minimum realistic ride < 4 hours
DURATION_MIN = 60
DURATION_MAX = 14400  # 4 hours

# Passenger count
PASSENGER_MIN = 1
PASSENGER_MAX = 9

# Vendor ID
VENDOR_MIN = 1
VENDOR_MAX = 2

# Store-and-forward flag (regex pattern for Pydantic)
STORE_AND_FWD_PATTERN = r"^[NY]$"

# Speed threshold (km/h) — trips implying > 150 km/h are sensor errors
SPEED_MAX = 150


# ---------------------------------------------------------------------------
# Feature Lists (shared by feature_pipeline.py & preprocess.py)
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "passenger_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "haversine_dist_km",
]

CATEGORICAL_COLS = [
    "vendor_id",
    "store_and_fwd_flag",
]

PASSTHROUGH_COLS = [
    "is_weekend",
    "is_rush_hour",
]


# ---------------------------------------------------------------------------
# Schema Metadata
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
FEATURE_SET_NAME = "nyc_eta_features_v1"
