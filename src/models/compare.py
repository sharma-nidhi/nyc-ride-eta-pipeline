# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import mlflow


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "NYC-ETA-Prediction"


def compare_runs(primary_metric: str = "mae", ascending: bool = True):
    """
    Read all completed runs from the current experiment and rank them.

    Parameters
    ----------
    primary_metric : str
        MLflow metric used for ranking (default: "mae").
    ascending : bool
        Set to False if lower is WORSE (e.g. r2). For mae/rmse keep True.

    Returns
    -------
    pd.DataFrame  – sorted run summary table.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
    )
    if not runs:
        print(f"No finished runs in experiment '{EXPERIMENT_NAME}'.")
        return None

    rows = []
    for r in runs:
        tags = r.data.tags or {}
        model_type = tags.get("model_type", tags.get("mlflow.runName", "unknown"))
        rows.append({
            "run_id": r.info.run_id[:8],
            "status": r.info.status,
            "run_name": tags.get("mlflow.runName", r.info.run_id[:8]),
            "model_type": model_type,
            "feature_count": r.data.params.get("feature_count", "?"),
            "dvc_slice": r.data.params.get("dvc_slice", "?"),
            "mae": r.data.metrics.get("mae", None),
            "rmse": r.data.metrics.get("rmse", None),
            "r2": r.data.metrics.get("r2", None),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.sort_values(by=primary_metric, ascending=ascending)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare registered ETA model runs")
    parser.add_argument("--metric", default="mae", help="Metric to rank by (mae, rmse, r2)")
    parser.add_argument("--ascending", action="store_true", default=True,
                        help="Sort ascending (lower is better). Omit for DESC (higher is better, e.g. r2).")
    args = parser.parse_args()

    df = compare_runs(primary_metric=args.metric, ascending=args.ascending)
    if df is not None:
        print(f"\n=== Model Comparison (sorted by {args.metric}) ===\n")
        print(df.to_string(index=False))
        print(f"\nBest run: {df.iloc[0]['run_id']}  "
              f"(MAE={df.iloc[0]['mae']:.2f}, RMSE={df.iloc[0]['rmse']:.2f}, "
              f"R2={df.iloc[0]['r2']:.4f})")
