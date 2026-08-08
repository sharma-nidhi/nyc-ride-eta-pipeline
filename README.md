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
```

### 2. Run Data Pipeline
```bash
python -m src.data.preprocess
```

### 3. Train Models
```bash
python -m src.models.train
```

### 4. Start API
```bash
uvicorn src.serving.api:app --reload
```

## 📈 Performance Summary
*(To be updated after Phase 2)*
- **Baseline RMSE:** TBD
- **Best Model RMSE:** TBD

## 🏗️ Architecture
Raw Data $\rightarrow$ `ingest.py` $\rightarrow$ `validate.py` $\rightarrow$ `preprocess.py` $\rightarrow$ `train.py` $\rightarrow$ MLflow $\rightarrow$ FastAPI
