# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import json
import pathlib
import mlflow
import logging

logger = logging.getLogger(__name__)

CHAMPION_PATH = pathlib.Path("models/champion.json")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "NYC-ETA-Prediction"


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
    best = min(runs, key=lambda r: r.data.metrics.get(primary_metric, float("inf")))

    tags = best.data.tags or {}
    champion = {
        "run_id": best.info.run_id,
        "run_name": tags.get("mlflow.runName", "unknown"),
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

    logger.info("Champion updated: run %s (MAE=%.2f, RMSE=%.2f, R2=%.4f)",
                champion["run_id"][:8],
                champion["metrics"]["mae"],
                champion["metrics"]["rmse"],
                champion["metrics"]["r2"])

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
