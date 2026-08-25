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

from src.contract import (
    SCHEMA_VERSION as FEATURE_VERSION,
    FEATURE_SET_NAME,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    PASSTHROUGH_COLS,
)


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
