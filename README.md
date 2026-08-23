# 🚗 nyc-ride-eta-pipeline

An end-to-end ML pipeline for predicting trip durations in NYC.

## 📌 Project Overview

This project implements a production-ready ML system including data validation, a robust feature engineering pipeline, model experimentation tracking with MLflow, and a REST API for real-time predictions.

## Team

- **Nidhi Sharma**
- **Kapil Chhabra**
- **Ronak Shah**

## Tech Stack

- **Language:** Python 3.10
- **Data Engineering:** Pandas, PyArrow (Parquet)
- **ML Pipeline:** Scikit-Learn
- **Experiment Tracking:** MLflow
- **Models:** XGBoost, LightGBM, CatBoost
- **Serving:** FastAPI, Uvicorn, Docker
- **Data Versioning:** DVC

## 🚀 Getting Started

### 1. Setup Environment

```bash
python -m venv .venv
.\venv\Scripts\activate
pip install -r requirements.txt
dvc init
```

### 2. Run Data Pipeline (with DVC Slicing)

The preprocessing script accepts an optional `--end-month` argument to simulate staged data ingestion (e.g., data arriving month-by-month).

**Full Dataset (Default):**

```bash
python -m src.data.preprocess
```

**Time-based Slicing:**

```bash
# Slice v1: Jan 2016 only
python -m src.data.preprocess --end-month 1

# Slice v2: Jan–Mar 2016
python -m src.data.preprocess --end-month 3
```

### 3. Version Data & Models (DVC)

After running the pipeline, snapshot the processed artifacts into DVC, then commit the pointer files to Git:

```bash
dvc add data/processed/X_train.parquet data/processed/y_train.parquet models/feature_pipeline.pkl
git add data/processed/*.dvc models/*.dvc data/contracts/feature_registry.json
git commit -m "feat(dvc): versioned dataset slice"
```

### 4. Train Models (Phase 2)

Train the baseline and all advanced models on the currently active dataset.
*Note: The training script uses a strictly chronological 80/20 split to prevent future data leakage (M2 rule). MLflow logs are stored in a lightweight SQLite database (`mlflow.db`), which is ignored by Git.*

**Train all models (Ridge, XGBoost, LightGBM, CatBoost):**

```bash
python -m src.models.train --model all
```

**Train a single model:**

```bash
python -m src.models.train --model ridge
python -m src.models.train --model xgboost
```

### 5. Compare Models & Select Champion

Compare all trained MLflow runs and promote the best model:

```bash
# Rank all runs by MAE (lower is better)
python -m src.models.compare --metric mae

# Rank by R2 (higher is better — use --no-ascending or omit --ascending)
python -m src.models.compare --metric r2 --ascending

# Promote the best run as the champion model
python -m src.models.registry
```

The champion metadata is saved to `models/champion.json` for Phase 3 serving.

### 6. View Experiment History

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

*(Open <http://127.0.0.1:5000> in your browser)*

### 7. Start API (Phase 3 — upcoming)

```bash
uvicorn src.serving.api:app --reload
```

## 📈 Performance Summary

| Model (Full Dataset) | MAE | RMSE | R² |
| ----- | --: | --: | --: |
| Ridge (Baseline) | 292.80s | 443.03s | 0.6148 |
| XGBoost | 245.11s | 384.68s | 0.7096 |
| LightGBM | 245.07s | 384.64s | 0.7097 |
| CatBoost | 246.17s | 386.22s | 0.7073 |
| **Champion** (LightGBM) | **245.07s** | **384.64s** | **0.7097** |

*Metrics from full-dataset 80/20 chronological split. Boosting models outperform Ridge baseline by ~16% on MAE. LightGBM and XGBoost are nearly tied; LightGBM edges ahead by 0.04s.*

## 🏗️ Architecture

```mermaid
graph LR
    raw["NYC.csv"] --> inject
    inject["ingest.py"] --> valid
    valid["validate.py"] --> fp
    fp["feature_pipeline.py"] --> train
    train["train.py"] --> mlflow
    mlflow["MLflow UI"] --> compare
    compare["compare.py"] --> registry
    registry["champion.json"] --> api
    api["FastAPI serving"] --> user["Client"]
```

Raw Data → `ingest.py` → `validate.py` → `preprocess.py` → `feature_pipeline.py` → `train.py` → MLflow → `compare.py` → `registry.py` → FastAPI
