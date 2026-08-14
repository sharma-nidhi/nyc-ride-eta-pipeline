"""
MLflow-tracked training + model comparison (Week 2 / M3).

PLACEHOLDER — implement training with full experiment tracking.

Plan:
  - Load features from the feature store; split (test_size, random_state from config).
  - Train baseline (LinearRegression) then gradient boosting (XGBoost/LightGBM).
  - Run several tracked experiments varying hyperparameters.
  - Log params + metrics (RMSLE, RMSE, MAE, R2) + model + feature_schema.json to MLflow.
  - Save the best model to models/eta-v1.joblib (random_state fixed = reproducible).

Metric note: NYC Taxi Trip Duration is scored by RMSLE — train/evaluate in log space.
"""
import mlflow


def main() -> None:
    mlflow.set_experiment("nyc_ride_eta")
    raise NotImplementedError("TODO Week 2: implement tracked training + comparison")


if __name__ == "__main__":
    main()
