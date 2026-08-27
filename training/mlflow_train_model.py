"""Training script with MLflow experiment tracking.

This preserves the previous preprocessing and model-evaluation behavior but
adds MLflow logging for parameters, provenance tags (git commit and data MD5),
metrics for each model, and the trained model artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_RAW = DATA_DIR / "raw" / "NYC_Train_raw.csv"
MODEL_STORE = BASE_DIR / "model_store"


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        out = out.decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("Loading raw CSV:", TRAIN_RAW)
    df = pd.read_csv(TRAIN_RAW, low_memory=False)

    # Basic cleaning and missing-value handling (same behavior as before)
    target = "IncidentGrade"
    df = df.dropna(subset=[target]).drop_duplicates()
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Temporal features
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Hour"] = df["Timestamp"].dt.hour
        df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
        df["Month"] = df["Timestamp"].dt.month
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df = df.drop(columns=["Timestamp"])

    # Drop identifier/high-cardinality columns (same set as before)
    drop_cols = [
        "Id", "OrgId", "IncidentId", "AlertId", "DeviceId", "Sha256",
        "IpAddress", "Url", "AccountSid", "AccountUpn", "AccountObjectId",
        "AccountName", "DeviceName", "NetworkMessageId", "EmailClusterId",
        "ApplicationId", "OAuthApplicationId", "ResourceIdName",
        "RegistryKey", "RegistryValueName", "RegistryValueData",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target])
    y = df[target]

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical_features:
        X_train[categorical_features] = encoder.fit_transform(X_train[categorical_features])
        X_val[categorical_features] = encoder.transform(X_val[categorical_features])
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # Models
    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_val)
    y_prob_baseline = baseline.predict_proba(X_val)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    y_prob_rf = rf.predict_proba(X_val)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(target_encoder.classes_), eval_metric="mlogloss",
        random_state=42, tree_method="hist",
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_xgb = xgb.predict(X_val)
    y_prob_xgb = xgb.predict_proba(X_val)

    # MLflow experiment and run
    mlflow.set_experiment("rep_incident_training")
    with mlflow.start_run(run_name="rf_xgb_comparison") as run:
        # params
        mlflow.log_params({
            "seed": 42,
            "test_size": 0.2,
            "rf_n_estimators": 100,
            "xgb_n_estimators": 300,
            "xgb_max_depth": 6,
        })

        # provenance tags
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_file": str(TRAIN_RAW),
            "data_md5": file_md5(TRAIN_RAW),
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        })

        # metrics - baseline
        mlflow.log_metric("baseline_accuracy", accuracy_score(y_val, y_pred_baseline))
        mlflow.log_metric("baseline_roc_auc", roc_auc_score(y_val, y_prob_baseline, multi_class="ovr"))
        mlflow.log_metric("baseline_macro_f1", f1_score(y_val, y_pred_baseline, average="macro"))

        # metrics - random forest
        mlflow.log_metric("rf_accuracy", accuracy_score(y_val, y_pred_rf))
        mlflow.log_metric("rf_roc_auc", roc_auc_score(y_val, y_prob_rf, multi_class="ovr"))
        mlflow.log_metric("rf_macro_f1", f1_score(y_val, y_pred_rf, average="macro"))

        # metrics - xgboost
        mlflow.log_metric("xgb_accuracy", accuracy_score(y_val, y_pred_xgb))
        mlflow.log_metric("xgb_roc_auc", roc_auc_score(y_val, y_prob_xgb, multi_class="ovr"))
        mlflow.log_metric("xgb_macro_f1", f1_score(y_val, y_pred_xgb, average="macro"))

        # log models as MLflow artifacts
        mlflow.sklearn.log_model(rf, "rf_model")
        # Use the dedicated xgboost flavor to avoid skops untrusted-type errors
        mlflow.xgboost.log_model(xgb, "xgb_model")

        # save the application artifacts locally as before
        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, MODEL_STORE / "rf_model.pkl")
        joblib.dump(encoder, MODEL_STORE / "feature_encoder.pkl")
        joblib.dump(target_encoder, MODEL_STORE / "target_encoder.pkl")
        with (MODEL_STORE / "feature_columns.json").open("w", encoding="utf-8") as fh:
            json.dump(X_train.columns.tolist(), fh)

        # print a concise run summary
        rf_auc = roc_auc_score(y_val, y_prob_rf, multi_class="ovr")
        xgb_auc = roc_auc_score(y_val, y_prob_xgb, multi_class="ovr")
        print(f"Run finished: id={run.info.run_id[:8]} | rf_auc={rf_auc:.4f} | xgb_auc={xgb_auc:.4f}")

    # display classification reports and plots locally (unchanged behavior)
    print("\nRandom Forest classification report:\n", classification_report(y_val, y_pred_rf, target_names=target_encoder.classes_))
    cm_rf = confusion_matrix(y_val, y_pred_rf)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    print("\nXGBoost classification report:\n", classification_report(y_val, y_pred_xgb, target_names=target_encoder.classes_))
    cm_xgb = confusion_matrix(y_val, y_pred_xgb)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")


if __name__ == "__main__":
    main()
