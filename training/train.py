import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

FEATURE_STORE = "data/processed/features.parquet"
SCHEMA_PATH = "models/feature_schema.json"
BEST_MODEL_PATH = "models/eta-v1.joblib"
EXPERIMENT = "nyc_ride_eta"
TRACKING_URI = "sqlite:///mlflow.db"  # DB backend (the plain-folder store is deprecated)
RANDOM_STATE = 42


def load_xy():
    df = pd.read_parquet(FEATURE_STORE)
    with open(SCHEMA_PATH) as f:
        feature_cols = json.load(f)
    X = df[feature_cols]
    y = np.log1p(df["trip_duration"])      
    return X, y, feature_cols


def evaluate(y_true_log, y_pred_log):
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_seconds = mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)
    return {"rmsle": rmsle, "mae_seconds": mae_seconds, "r2": r2}


def train_and_log(name, model, X_train, X_test, y_train, y_test):
    """Fit a model, evaluate it, and log everything to MLflow as one run."""
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(SCHEMA_PATH)

    print(f"{name:20s}  RMSLE={metrics['rmsle']:.4f}  "
          f"MAE={metrics['mae_seconds']:.0f}s  R2={metrics['r2']:.4f}")
    return metrics, model


if __name__ == "__main__":
    X, y, feature_cols = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} | Features: {len(feature_cols)}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # --- Models to train: baseline + two gradient-boosting configs ---
    candidates = [
        ("linear_regression", LinearRegression()),
        ("hgb_it200_d6_lr0.10",
         HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                       learning_rate=0.10, random_state=RANDOM_STATE)),
        ("hgb_it400_d8_lr0.05",
         HistGradientBoostingRegressor(max_iter=400, max_depth=8,
                                       learning_rate=0.05, random_state=RANDOM_STATE)),
    ]

    results = []
    for name, model in candidates:
        metrics, fitted = train_and_log(name, model, X_train, X_test, y_train, y_test)
        results.append((name, metrics, fitted))

    # --- Pick the winner (lowest RMSLE) and save it ---
    best_name, best_metrics, best_model = min(results, key=lambda r: r[1]["rmsle"])
    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n=== Comparison (lower RMSLE is better) ===")
    for name, metrics, _ in results:
        flag = "  <-- BEST" if name == best_name else ""
        print(f"  {name:22s}  RMSLE={metrics['rmsle']:.4f}  "
              f"MAE={metrics['mae_seconds']:.0f}s{flag}")
    print(f"\nSaved best model ({best_name}) -> {BEST_MODEL_PATH}")
