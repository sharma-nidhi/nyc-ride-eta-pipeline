# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Model Registry
==============
Automatically selects the best MLflow run as champion based on primary
metric (MAE), serializes the winning model to `serving/model.pkl`, and
writes `champion.json` with run metadata, parameters, and metrics."""

import json
import pathlib
import mlflow
import logging
import shutil

logger = logging.getLogger(__name__)

CHAMPION_PATH = pathlib.Path("models/champion.json")
MODEL_OUTPUT_DIR = pathlib.Path("models/artifacts")
SERVING_MODEL_PATH = pathlib.Path("models/serving/model.pkl")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "NYC-ETA-Prediction"
KNOWN_MODELS = {"ridge", "xgboost", "lightgbm", "catboost"}


def promote_champion(primary_metric: str = "mae"):
    """
    Find the best run in the experiment and promote it as the champion model.

    Parameters
    ----------
    primary_metric : str
        Metric to optimise (lower is better: mae/rmse, default "mae").

    Returns
    -------
    dict  – champion metadata written to champion.json.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(experiment_ids=[exp.experiment_id],
                              filter_string="status = 'FINISHED'")
    if not runs:
        raise RuntimeError(f"No finished runs in '{EXPERIMENT_NAME}'.")

    # Sort by primary metric (lower is better for mae/rmse).
    # Includes all runs (default + tuned) — the best model wins regardless of how it was trained.
    best = min(runs, key=lambda r: r.data.metrics.get(primary_metric, float("inf")))

    tags = best.data.tags or {}
    run_name = tags.get("mlflow.runName", "unknown")
    model_type = tags.get("model_type", str(run_name).lower())
    if model_type not in KNOWN_MODELS:
        lowered = str(run_name).lower()
        model_type = lowered.split("-")[0]
    if model_type not in KNOWN_MODELS:
        raise RuntimeError(
            f"Cannot resolve model type from run '{run_name}'. "
            f"Expected one of {sorted(KNOWN_MODELS)}."
        )

    champion = {
        "run_id": best.info.run_id,
        "run_name": run_name,
        "model_type": model_type,
        "status": best.info.status,
        "primary_metric": primary_metric,
        "metrics": {
            "mae": best.data.metrics.get("mae"),
            "rmse": best.data.metrics.get("rmse"),
            "r2": best.data.metrics.get("r2"),
        },
        "feature_count": best.data.params.get("feature_count", "?"),
        "dvc_slice": best.data.params.get("dvc_slice", "?"),
        "promoted_at": None,  # placeholder; could use iso timestamp
    }

    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_PATH, "w", encoding="utf-8") as f:
        json.dump(champion, f, indent=2)

    # Export a stable serving artifact path so deployment images only need one model file.
    source_model_path = MODEL_OUTPUT_DIR / f"{champion['model_type']}.pkl"
    if not source_model_path.exists():
        raise FileNotFoundError(
            f"Champion artifact '{source_model_path}' not found. "
            "Ensure training artifacts exist before promoting."
        )
    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_path, SERVING_MODEL_PATH)

    logger.info("Champion updated: run %s (MAE=%.2f, RMSE=%.2f, R2=%.4f)",
                champion["run_id"][:8],
                champion["metrics"]["mae"],
                champion["metrics"]["rmse"],
                champion["metrics"]["r2"])
    logger.info("Serving artifact exported to %s", SERVING_MODEL_PATH)

    return champion


def get_champion() -> dict | None:
    """Load current champion from disk, or None if not yet set."""
    if not CHAMPION_PATH.exists():
        return None
    with open(CHAMPION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Promote best model run as champion")
    parser.add_argument("--metric", default="mae",
                        help="Metric to optimise (default: mae)")
    args = parser.parse_args()

    champion = promote_champion(args.metric)
    print(f"\n=== Champion Model ===\n")
    print(f"  Run  : {champion['run_id']}")
    print(f"  Name : {champion['run_name']}")
    print(f"  MAE  : {champion['metrics']['mae']:.2f}s")
    print(f"  RMSE : {champion['metrics']['rmse']:.2f}s")
    print(f"  R2   : {champion['metrics']['r2']:.4f}")
    print(f"  Slice: {champion['dvc_slice']}")
    print(f"\nArtifact saved to {CHAMPION_PATH}")
    print(f"Serving model exported to {SERVING_MODEL_PATH}")
