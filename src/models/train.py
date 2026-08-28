# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pathlib
import logging
import joblib
import json
import hashlib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from mlflow.models import infer_signature
from typing import Tuple, Dict, Any

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.reproducibility import DEFAULT_RANDOM_SEED, seed_everything
from src.models.optuna_tuner import tune_model, TUNABLE_MODELS

logger = logging.getLogger(__name__)

# Paths
PROCESSED_X = pathlib.Path("data/processed/X_train.parquet")
PROCESSED_Y = pathlib.Path("data/processed/y_train.parquet")
MODEL_OUTPUT_DIR = pathlib.Path("models/artifacts")
FEATURE_REGISTRY_PATH = pathlib.Path("data/contracts/feature_registry.json")
VALIDATION_REPORT_PATH = pathlib.Path("data/validation_report.json")

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT = "NYC-ETA-Prediction"


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def train_ridge(X_train, y_train, X_val, y_val, seed: int = DEFAULT_RANDOM_SEED):
    """Baseline Ridge Regression model."""
    model = Ridge(alpha=1.0)
    return model, "Ridge", {"alpha": 1.0, "random_seed": seed}


def train_xgboost(X_train, y_train, X_val, y_val, seed: int = DEFAULT_RANDOM_SEED):
    """XGBoost model."""
    import xgboost as xgb  # noqa: F811
    model = xgb.XGBRegressor(
        n_estimators=123,
        max_depth=8,
        learning_rate=0.0605,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    return model, "XGBoost", {
        "n_estimators": 123,
        "max_depth": 8,
        "learning_rate": 0.0605,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_seed": seed,
    }


def train_lightgbm(X_train, y_train, X_val, y_val, seed: int = DEFAULT_RANDOM_SEED):
    """LightGBM model."""
    import lightgbm as lgb  # noqa: F811
    model = lgb.LGBMRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.0896,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    return model, "LightGBM", {
        "n_estimators": 250,
        "max_depth": 8,
        "learning_rate": 0.0896,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_seed": seed,
    }


def train_catboost(X_train, y_train, X_val, y_val, seed: int = DEFAULT_RANDOM_SEED):
    """CatBoost model."""
    import catboost as cb  # noqa: F811
    model = cb.CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_seed=seed,
        verbose=0,
    )
    return model, "CatBoost", {
        "iterations": 300,
        "depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "random_seed": seed,
    }


TRAINERS = {
    "ridge": train_ridge,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
}

# Baseline params used only for tune-summary comparison output.
TUNING_START_PARAMS = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def chronologic_split(X, y, train_ratio=0.8):
    """Chronological 80/20 split to prevent future data leakage."""
    idx = int(len(X) * train_ratio)
    return X.iloc[:idx], X.iloc[idx:], y.iloc[:idx], y.iloc[idx:]


def log_feature_contract(schema: dict):
    """Log feature contract metadata and artifact."""
    feature_names = [col["name"] for col in schema.get("feature_columns", [])]
    feature_hash = hashlib.sha256(
        json.dumps(feature_names, sort_keys=True).encode("utf-8")
    ).hexdigest()
    mlflow.log_params({
        "feature_set": schema.get("feature_set", "unknown"),
        "schema_version": schema.get("schema_version", "unknown"),
        "feature_count": len(feature_names),
        "dvc_slice": schema.get("dvc_slice", "unknown"),
        "feature_list_sha256": feature_hash,
    })
    mlflow.log_artifact(str(FEATURE_REGISTRY_PATH), "contracts")


def log_validation_report():
    """Attach validation report to MLflow run if present."""
    if VALIDATION_REPORT_PATH.exists():
        mlflow.log_artifact(str(VALIDATION_REPORT_PATH), "validation")


# ---------------------------------------------------------------------------
# Main training entry-points
# ---------------------------------------------------------------------------

def run_training(model_type: str = "ridge",
                 X: pd.DataFrame | None = None,
                 y: pd.Series | None = None,
                 seed: int = DEFAULT_RANDOM_SEED) -> Tuple:
    """Train a single model type with full MLflow tracking."""
    if X is None or y is None:
        X = pd.read_parquet(PROCESSED_X)
        y = pd.read_parquet(PROCESSED_Y).squeeze()

    X_train, X_val, y_train, y_val = chronologic_split(X, y)

    trainer = TRAINERS.get(model_type)
    if trainer is None:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(TRAINERS.keys())}")

    logger.info("Training %s model ...", model_type.upper())
    seed_everything(seed)
    model, friendly_name, params = trainer(X_train, y_train, X_val, y_val, seed=seed)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name=model_type.upper()):
        mlflow.set_tag("model_type", model_type)
        # Data params
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("x_null_fraction", float(X.isna().mean().mean()))
        mlflow.log_param("global_random_seed", seed)

        # Model params
        mlflow.log_params(params)

        # Feature contract + validation
        if FEATURE_REGISTRY_PATH.exists():
            with open(FEATURE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
            log_feature_contract(schema)
        log_validation_report()

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        logger.info("%s Metrics: MAE=%.2f, RMSE=%.2f, R2=%.4f",
                     friendly_name, mae, rmse, r2)

        # Log model using each library's native MLflow flavor to avoid skops type issues
        signature = infer_signature(X_val, predictions)
        artifact_name = f"{model_type}_model"
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, name=artifact_name, signature=signature)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, name=artifact_name, signature=signature)
        elif model_type == "catboost":
            mlflow.catboost.log_model(model, name=artifact_name, signature=signature)
        else:
            mlflow.sklearn.log_model(model, name=artifact_name, signature=signature)

        # Save local binary for DVC
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_OUTPUT_DIR / f"{model_type}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

    return model, {"mae": mae, "rmse": rmse, "r2": r2, "params": params}


def run_all_models(X: pd.DataFrame | None = None,
                   y: pd.Series | None = None,
                   seed: int = DEFAULT_RANDOM_SEED) -> dict[str, dict]:
    """Train all available models and return {model_type: metrics + params}."""
    results = {}
    for model_type in TRAINERS:
        _, metrics = run_training(model_type, X, y, seed=seed)
        results[model_type] = metrics
    return results


def run_training_with_tuning(model_type: str,
                             X: pd.DataFrame | None = None,
                             y: pd.Series | None = None,
                             seed: int = DEFAULT_RANDOM_SEED,
                             n_trials: int = 20) -> Tuple:
    """Tune a model with Optuna, then train & log using the best params."""
    if model_type not in TUNABLE_MODELS:
        return run_training(model_type, X, y, seed=seed)

    if X is None or y is None:
        X = pd.read_parquet(PROCESSED_X)
        y = pd.read_parquet(PROCESSED_Y).squeeze()

    X_train, X_val, y_train, y_val = chronologic_split(X, y)
    seed_everything(seed)
    best_params, _best_val_mae = tune_model(
        model_type, X_train, y_train, X_val, y_val, n_trials=n_trials
    )

    # Build model using Optuna-best params.
    import xgboost as xgb
    import lightgbm as lgb
    friendly_name = model_type.upper()
    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
        params = {
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_seed": seed,
        }
    else:  # lightgbm
        model = lgb.LGBMRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        params = {
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_seed": seed,
        }

    # Train
    if model_type == "xgboost":
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)
    logger.info("%s Metrics: MAE=%.2f, RMSE=%.2f, R2=%.4f",
                friendly_name, mae, rmse, r2)

    # MLflow tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=model_type.upper()):
        mlflow.set_tag("model_type", model_type)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("x_null_fraction", float(X.isna().mean().mean()))
        mlflow.log_param("global_random_seed", seed)
        mlflow.log_params(params)
        mlflow.log_param("tune_trials", n_trials)
        mlflow.log_param("tune_framework", "optuna")

        if FEATURE_REGISTRY_PATH.exists():
            with open(FEATURE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
            log_feature_contract(schema)
        log_validation_report()

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

        signature = infer_signature(X_val, predictions)
        artifact_name = f"{model_type}_model"
        if model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, name=artifact_name, signature=signature)
        else:
            mlflow.xgboost.log_model(model, name=artifact_name, signature=signature)

        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_OUTPUT_DIR / f"{model_type}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

    return model, {"mae": mae, "rmse": rmse, "r2": r2, "params": params}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train ETA prediction models")
    parser.add_argument(
        "--model",
        choices=list(TRAINERS.keys()) + ["all"],
        default="all",
        help="Model type to train (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Global random seed for reproducible training",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning for LightGBM and XGBoost (20 trials each)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=20,
        help="Number of Optuna trials per model (default: 20)",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading processed data ...")
        X = pd.read_parquet(PROCESSED_X)
        y = pd.read_parquet(PROCESSED_Y).squeeze()

        if args.model == "all":
            if args.tune:
                results = _train_all_with_tuning(X, y, seed=args.seed, n_trials=args.tune_trials)
            else:
                results = run_all_models(X, y, seed=args.seed)
            print("\n=== Training Summary ===")
            for name, m in results.items():
                params_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in m.get("params", {}).items()
                )
                print(f"  {name:12s}  MAE: {m['mae']:8.2f}s  RMSE: {m['rmse']:8.2f}s  R2: {m['r2']:.4f}")
                print(f"  {'':12s}  {params_str}")
        else:
            if args.tune and args.model in TUNABLE_MODELS:
                _, metrics = run_training_with_tuning(
                    args.model,
                    X,
                    y,
                    seed=args.seed,
                    n_trials=args.tune_trials,
                )
            else:
                _, metrics = run_training(args.model, X, y, seed=args.seed)
            params_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.get("params", {}).items()
            )
            print(f"\n--- {args.model.upper()} Training Complete ---")
            print(f"MAE:  {metrics['mae']:.2f}s")
            print(f"RMSE: {metrics['rmse']:.2f}s")
            print(f"R2:   {metrics['r2']:.4f}")
            print(f"Params: {params_str}")

        print("\nExperiment tracked in MLflow. Run:")
        print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Tuning-aware training
# ---------------------------------------------------------------------------

def _train_all_with_tuning(X, y, seed: int = DEFAULT_RANDOM_SEED,
                           n_trials: int = 20) -> dict[str, dict]:
    """Train all 4 models. LightGBM and XGBoost are tuned with Optuna; others use defaults."""
    X_train, X_val, y_train, y_val = chronologic_split(X, y)
    results = {}
    print("\n" + "=" * 60)
    print(f"Optuna Tuning ({n_trials} trials per model)")
    print("=" * 60)

    for model_type in TRAINERS:
        if model_type in TUNABLE_MODELS:
            logger.info("Tuning %s ...", model_type.upper())
            seed_everything(seed)
            best_params, _best_val_mae = tune_model(
                model_type, X_train, y_train, X_val, y_val, n_trials=n_trials
            )

            # Show baseline vs Optuna best params side-by-side.
            comparison_params = TUNING_START_PARAMS.get(model_type, {})
            print(f"\n--- {model_type.upper()} Params (baseline → optuna-best) ---")
            for k in ("n_estimators", "max_depth", "learning_rate"):
                dv = comparison_params.get(k, "?")
                tv = best_params.get(k, "?")
                if k == "learning_rate":
                    print(f"  {k:20s}: {dv:>10.4f} → {tv:>10.4f}")
                else:
                    print(f"  {k:20s}: {dv:>10} → {tv:>10}")

            # Build model using Optuna-best params.
            import xgboost as xgb
            import lightgbm as lgb
            if model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=best_params["n_estimators"],
                    max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    n_jobs=-1,
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=best_params["n_estimators"],
                    max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            params = {
                "n_estimators": best_params["n_estimators"],
                "max_depth": best_params["max_depth"],
                "learning_rate": best_params["learning_rate"],
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_seed": seed,
            }
            friendly_name = model_type.upper()

            # Train
            if model_type == "xgboost":
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
        else:
            logger.info("Training %s (default) ...", model_type.upper())
            seed_everything(seed)
            model, friendly_name, params = TRAINERS[model_type](X_train, y_train, X_val, y_val, seed=seed)
            model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        logger.info("%s Metrics: MAE=%.2f, RMSE=%.2f, R2=%.4f", friendly_name, mae, rmse, r2)

        # MLflow tracking
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=model_type.upper()):
            mlflow.set_tag("model_type", model_type)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("x_null_fraction", float(X.isna().mean().mean()))
            mlflow.log_param("global_random_seed", seed)
            mlflow.log_params(params)
            if model_type in TUNABLE_MODELS:
                mlflow.log_param("tune_trials", n_trials)
                mlflow.log_param("tune_framework", "optuna")
            if FEATURE_REGISTRY_PATH.exists():
                with open(FEATURE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                log_feature_contract(schema)
            log_validation_report()
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

            sig = infer_signature(X_val, preds)
            artifact_name = f"{model_type}_model"
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, name=artifact_name, signature=sig)
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, name=artifact_name, signature=sig)
            elif model_type == "catboost":
                mlflow.catboost.log_model(model, name=artifact_name, signature=sig)
            else:
                mlflow.sklearn.log_model(model, name=artifact_name, signature=sig)

        # Save local .pkl
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_OUTPUT_DIR / f"{model_type}.pkl"
        joblib.dump(model, model_path)
        results[model_type] = {"mae": mae, "rmse": rmse, "r2": r2, "params": params}

    return results


if __name__ == "__main__":
    cli_main()
