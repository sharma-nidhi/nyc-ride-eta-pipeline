# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Projection Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import json
import joblib
import pathlib
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Paths (relative to project root)
CHAMPION_PATH = pathlib.Path("models/champion.json")
MODEL_OUTPUT_DIR = pathlib.Path("models/artifacts")
SERVING_MODEL_PATH = pathlib.Path("models/serving/model.pkl")
FEATURE_PIPELINE_PATH = pathlib.Path("models/feature_pipeline.pkl")


def load_model() -> Tuple:
    """Read champion.json and load the corresponding model + feature pipeline.

    Returns
    -------
    (model, feature_pipeline)
        - model: fitted estimator
        - feature_pipeline: sklearn Pipeline for raw → feature transforms
    """
    with open(CHAMPION_PATH, "r", encoding="utf-8") as f:
        champion = json.load(f)

    model_type = champion["run_name"].lower()  # e.g. "lightgbm"

    # Primary path for deployment images: a single promoted champion artifact.
    model_path = SERVING_MODEL_PATH
    if not model_path.exists():
        # Backward-compatible fallback for local/dev environments.
        model_path = MODEL_OUTPUT_DIR / f"{model_type}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{SERVING_MODEL_PATH}' or '{MODEL_OUTPUT_DIR / f'{model_type}.pkl'}'. "
            "Run `python -m src.models.registry` after training to export champion artifact."
        )

    model = joblib.load(model_path)
    feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)

    logger.info("Loaded %s model (run %s, MAE=%.2f)",
                 model_type, champion["run_id"][:8], champion["metrics"]["mae"])

    return model, feature_pipeline


def get_champion_info() -> dict:
    """Load champion metadata for the /model-info endpoint."""
    with open(CHAMPION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
