# Model Comparison Report — NYC Ride ETA (Week 2 / M3)

**Experiment:** `nyc_ride_eta` (MLflow) · **Best model saved:** `models/eta-v1.joblib`

## Setup
- **Data:** `data/processed/features.parquet` — 1,450,674 cleaned trips, 9 features
  (hour, day_of_week, is_weekend, month, haversine_km, manhattan_km, passenger_count,
  vendor_id, store_and_fwd).
- **Target:** `trip_duration` (seconds), trained on `log1p(trip_duration)`.
- **Primary metric:** **RMSLE** (Root Mean Squared Log Error — the Kaggle metric). Because we
  train in log space, RMSE in log space *equals* RMSLE. We also report **MAE in seconds**
  (human-readable) and **R²**.
- **Split:** 80 / 20 train/test, `random_state=42` (fixed → reproducible).
- **Tracking:** MLflow, SQLite backend (`sqlite:///mlflow.db`); each run logs params, metrics,
  the model artifact, and `feature_schema.json`.

## Results
| Run | Key hyperparameters | RMSLE ↓ | MAE (s) ↓ | R² ↑ |
|-----|--------------------|:-------:|:---------:|:----:|
| linear_regression | — (baseline) | 0.5432 | 355 | 0.459 |
| hgb_it200_d6_lr0.10 | max_iter=200, max_depth=6, lr=0.10 | 0.3970 | 219 | 0.711 |
| **hgb_it400_d8_lr0.05** | **max_iter=400, max_depth=8, lr=0.05** | **0.3967** | **219** | **0.712** |

*(See `docs/images/mlflow_compare.png` for the MLflow comparison screenshot.)*

## Analysis & model selection
- **Gradient boosting decisively beats the linear baseline:** RMSLE improves ~27%
  (0.543 → 0.397), average error drops from ~5.9 min to ~3.6 min, and R² rises from 0.46 to
  0.71. Trip time is non-linear in distance/time-of-day, which the boosted trees capture and a
  linear model cannot.
- **The two boosting configs are effectively tied:** 0.3967 vs 0.3970 RMSLE — a 0.0003
  difference despite the winner using 2× the iterations and greater depth. The extra capacity
  buys essentially nothing here.
- **Selected model:** `hgb_it400_d8_lr0.05` (lowest RMSLE), serialized to `models/eta-v1.joblib`.
- **Engineering note / trade-off:** because the lighter `hgb_it200_d6` matches accuracy at roughly
  half the training and inference cost (and with lower overfitting risk), it is an equally
  defensible — arguably better — production choice. If serving latency or cost becomes a concern
  in Week 3, switching to the smaller model is a one-line change with no measurable accuracy loss.

## Reproducibility
- Fixed `random_state=42` for both the split and the models → re-running `python training/train.py`
  produces identical metrics (verified: 0.5432 / 0.3970 / 0.3967).
- Every run's full configuration is logged in MLflow; any run can be reproduced from its logged
  params on the same dataset version (`data-v1`).
- **Reproduce from scratch:** `git checkout` → `dvc pull` → `python data/validate.py` →
  `python features/build_features.py` → `python training/train.py`.

## Next (Week 3 / M4)
Package `models/eta-v1.joblib` behind a FastAPI `/predict` endpoint that reuses
`features/build_features.py` so serving matches training exactly.
