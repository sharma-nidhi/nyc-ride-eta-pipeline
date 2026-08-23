# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Feature Definitions & Metadata
===============================
This file serves as the "Source of Truth" for the feature contract. It defines
the **output schema** expected by the trained model. It prevents Train-Serving
skew by decoupling feature configuration from pipeline implementation.

In a production system, this maps directly to the **Feature Registry** inside
a Feature Store (e.g., Feast, Tecton, or Hopsworks).
"""

FEATURE_VERSION = "1.0.0"
FEATURE_SET_NAME = "nyc_eta_features_v1"

# ---------------------------------------------------------------------------
# OUTPUT FEATURE CONTRACT
# ---------------------------------------------------------------------------
# These lists define the exact columns the trained model expects in its input
# DataFrame. They are shared by:
#   1. The Feature Pipeline  (to know which sklearn transforms to apply)
#   2. The Preprocess script (to build the JSON registry)
#   3. The FastAPI service   (to validate incoming JSON requests)
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


def get_feature_schema() -> dict:
    """
    Declarative feature registry (Lightweight Feature Store contract).

    Returns a JSON-compatible dictionary describing every feature in the model's
    output schema, including its expected dtype, transform, and category.

    This is automatically dumped to data/contracts/feature_registry.json by the
    preprocessing pipeline after every DVC slice.
    """
    return {
        "schema_version": FEATURE_VERSION,
        "feature_set": FEATURE_SET_NAME,
        "feature_columns": (
            [
                {
                    "name": c,
                    "type": "float32",
                    "transform": "RobustScaler",
                    "category": "numeric",
                }
                for c in NUMERIC_COLS
            ]
            + [
                {
                    "name": c,
                    "type": "category",
                    "transform": "OrdinalEncoder",
                    "category": "categorical",
                }
                for c in CATEGORICAL_COLS
            ]
            + [
                {
                    "name": c,
                    "type": "float32",
                    "transform": "None (Passthrough)",
                    "category": "passthrough",
                }
                for c in PASSTHROUGH_COLS
            ]
        ),
    }
