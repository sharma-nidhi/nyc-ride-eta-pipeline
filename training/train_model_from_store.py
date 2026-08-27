"""Train REP models exclusively from the persisted feature store."""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_STORE_DB = BASE_DIR / "feature_store" / "feature_store.db"
FEATURE_STORE_TABLE = "incident_features"
MODEL_STORE = BASE_DIR / "model_store"
TARGET = "IncidentGrade"


def load_features_from_store(
    database_path: Path = FEATURE_STORE_DB,
) -> pd.DataFrame:
    """Load the complete prepared feature table from SQLite."""
    if not database_path.exists():
        raise FileNotFoundError(
            f"Feature store not found: {database_path}. "
            "Run features/build_feature.py first."
        )

    with sqlite3.connect(database_path) as connection:
        feature_table = pd.read_sql_query(
            f"SELECT * FROM {FEATURE_STORE_TABLE}", connection
        )

    if TARGET not in feature_table.columns:
        raise ValueError(
            f"Feature-store table {FEATURE_STORE_TABLE!r} must contain {TARGET!r}"
        )

    print(f"Loaded {len(feature_table)} rows from feature store")
    print(f"Feature schema: {list(feature_table.drop(columns=[TARGET]).columns)}")
    return feature_table


print("Loading features from feature store:", FEATURE_STORE_DB)
feature_df = load_features_from_store()

# The feature builder has already performed cleaning, timestamp extraction,
# and identifier removal.
X = feature_df.drop(columns=[TARGET])
y = feature_df[TARGET]
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Target distribution:")
print(y.value_counts())

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print("Target encoding:")
for label, number in zip(
    target_encoder.classes_,
    target_encoder.transform(target_encoder.classes_),
):
    print(label, "=", number)

# Keep the same validation split and random state as the original workflow.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)
X_train = X_train.copy()
X_val = X_val.copy()
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# Keep the same categorical encoding behavior as the original workflow.
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
print("Categorical features:", categorical_features)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
if categorical_features:
    X_train[categorical_features] = encoder.fit_transform(
        X_train[categorical_features]
    )
    X_val[categorical_features] = encoder.transform(X_val[categorical_features])
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_val:", X_val.isnull().sum().sum())


# Train Baseline model
baseline_model = DummyClassifier(strategy="most_frequent", random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_val)
y_prod_baseline = baseline_model.predict_proba(X_val)
print("Baseline Accuracy:", accuracy_score(y_val, y_pred_baseline))
print("ROC_AUC:", roc_auc_score(y_val, y_prod_baseline, multi_class="ovr"))
print("Baseline Macro F1:", f1_score(y_val, y_pred_baseline, average="macro"))


# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
y_prod_rf = rf_model.predict_proba(X_val)
print("Random Forest training completed.")


# Evaluate Random Forest model
rf_accuracy = accuracy_score(y_val, y_pred_rf)
rf_roc_score = roc_auc_score(y_val, y_prod_rf, multi_class="ovr")
rf_macro_f1 = f1_score(y_val, y_pred_rf, average="macro")
print("Accuracy:", rf_accuracy)
print("ROC_AUC:", rf_roc_score)
print("Macro F1 Score:", rf_macro_f1)
print("\nClassification Report:")
print(classification_report(y_val, y_pred_rf, target_names=target_encoder.classes_))

cm_rf = confusion_matrix(y_val, y_pred_rf)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")


# Random Forest feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": rf_model.feature_importances_}
).sort_values("Importance", ascending=False)
top_features = feature_importance.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 15 Important Features - Random Forest")
plt.gca().invert_yaxis()


# Train XGBoost model
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(target_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist",
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_xgb = xgb_model.predict(X_val)
y_prod_xgb = xgb_model.predict_proba(X_val)
print("XGBoost training completed.")


# Evaluate XGBoost model
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)
xgb_roc_score = roc_auc_score(y_val, y_prod_xgb, multi_class="ovr")
xgb_macro_f1 = f1_score(y_val, y_pred_xgb, average="macro")
print("Accuracy:", xgb_accuracy)
print("ROC_AUC:", xgb_roc_score)
print("Macro F1 Score:", xgb_macro_f1)
print("\nClassification Report:")
print(classification_report(y_val, y_pred_xgb, target_names=target_encoder.classes_))

cm_xgb = confusion_matrix(y_val, y_pred_xgb)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
)
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")


# Model comparison
model_results = pd.DataFrame(
    {
        "Model": ["Baseline", "Random Forest", "XGBoost"],
        "Accuracy": [
            accuracy_score(y_val, y_pred_baseline),
            accuracy_score(y_val, y_pred_rf),
            accuracy_score(y_val, y_pred_xgb),
        ],
        "ROC_AUC": [
            roc_auc_score(y_val, y_prod_baseline, multi_class="ovr"),
            roc_auc_score(y_val, y_prod_rf, multi_class="ovr"),
            roc_auc_score(y_val, y_prod_xgb, multi_class="ovr"),
        ],
        "Macro F1 Score": [
            f1_score(y_val, y_pred_baseline, average="macro"),
            f1_score(y_val, y_pred_rf, average="macro"),
            f1_score(y_val, y_pred_xgb, average="macro"),
        ],
    }
)
print(model_results)

plt.figure(figsize=(8, 5))
sns.barplot(data=model_results, x="Model", y="Macro F1 Score")
plt.title("Model Comparison Based on Macro F1 Score")
plt.xlabel("Model")
plt.ylabel("Macro F1 Score")


# Save artifacts. Random Forest remains the model used by the application.
MODEL_STORE.mkdir(parents=True, exist_ok=True)
joblib.dump(rf_model, MODEL_STORE / "rf_model.pkl")
joblib.dump(encoder, MODEL_STORE / "feature_encoder.pkl")
joblib.dump(target_encoder, MODEL_STORE / "target_encoder.pkl")

# Derive the schema from the feature store rather than hardcoding it.
with (MODEL_STORE / "feature_columns.json").open("w", encoding="utf-8") as file:
    json.dump(X_train.columns.tolist(), file)

print("Model retrained from feature store")
print(f"Feature schema: {list(X_train.columns)}")
