# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""Optuna-based hyperparameter tuning for LightGBM and XGBoost models."""

import optuna
import mlflow
import logging
from sklearn.metrics import mean_absolute_error

from src.reproducibility import DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "NYC-ETA-Prediction"
N_TRIALS = 20  # Keep minimal for speed


# -----------------------------------------------------------------------
# LightGBM Tuner
# -----------------------------------------------------------------------

def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = N_TRIALS):
    """Find best LightGBM hyperparameters using Optuna."""
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "random_state": DEFAULT_RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(
        direction="minimize",
        study_name="lightgbm_tuning",
        sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("LightGBM Best MAE: %.2f | Params: %s", study.best_value, study.best_params)
    return study.best_params, study.best_value


# -----------------------------------------------------------------------
# XGBoost Tuner
# -----------------------------------------------------------------------

def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int = N_TRIALS):
    """Find best XGBoost hyperparameters using Optuna."""
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "random_state": DEFAULT_RANDOM_SEED,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(
        direction="minimize",
        study_name="xgboost_tuning",
        sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("XGBoost Best MAE: %.2f | Params: %s", study.best_value, study.best_params)
    return study.best_params, study.best_value


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

TUNABLE_MODELS = {
    "lightgbm": optimize_lightgbm,
    "xgboost": optimize_xgboost,
}

def tune_model(
    model_type: str,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = N_TRIALS,
) -> tuple[dict, float]:
    """Tune a model and return (best_params, best_mae)."""
    tuner = TUNABLE_MODELS.get(model_type)
    if tuner is None:
        raise ValueError(
            f"Tuning not configured for '{model_type}'. Available: {list(TUNABLE_MODELS.keys())}"
        )
    return tuner(X_train, y_train, X_val, y_val, n_trials=n_trials)
