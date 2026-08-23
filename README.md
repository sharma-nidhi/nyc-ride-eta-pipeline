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

### 4. Train Models

Train the baseline model on the currently active dataset.
*Note: The training script uses a strictly chronological 80/20 split to prevent future data leakage (M2 rule). MLflow logs are stored in a lightweight SQLite database (`mlflow.db`), which is ignored by Git.*

```bash
python -m src.models.train
```

To view your experiment history, metrics, and artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

*(Open <http://127.0.0.1:5000> in your browser)*

### 5. Start API (Phase 3)

```bash
uvicorn src.serving.api:app --reload
```

## 📈 Performance Summary

*(To be updated after Phase 2)*

- **Baseline RMSE:** TBD
- **Best Model RMSE:** TBD

## 🏗️ Architecture

Raw Data $\rightarrow$ `ingest.py` $\rightarrow$ `validate.py` $\rightarrow$ `preprocess.py` $\rightarrow$ `train.py` $\rightarrow$ MLflow $\rightarrow$ FastAPI
