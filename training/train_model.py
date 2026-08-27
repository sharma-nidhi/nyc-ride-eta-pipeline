# Importing libraries for EDA
import json
from pathlib import Path

import os
from xml.parsers.expat import model
import joblib
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
# Suppress common, non-actionable warnings from libraries (Future/User warnings)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

## Loading the dataset ##

train_path = DATA_DIR /"raw" / "NYC_Train_raw.csv"
print("Loading dataset from:", train_path)
try:
    train_df = pd.read_csv(train_path, nrows=300000, low_memory=False)
except Exception as e:
    raise RuntimeError(f"Failed to load training data from {train_path}: {e}")
print("Train shape:", train_df.shape)
train_df.head()

# Summary statistics and data types
print("Summary Statistics:")
train_df.describe()
print("\nDataset Information:")
train_df.info()

# Analyse missing values
train_clean = train_df.copy()
missing = train_clean.isnull().sum()
missing_table = pd.DataFrame({"Column": missing.index, "Missing_Count": missing.values})
missing_table = missing_table[missing_table["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)
print("="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
print(missing_table)


# Handle missing values

target = "IncidentGrade"
train_clean = train_df.copy()
train_clean = train_clean.dropna(subset=[target])
print("Shape after removing missing IncidentGrade:", train_clean.shape)
for col in train_clean.columns:
    if col == target:
        continue
    if pd.api.types.is_numeric_dtype(train_clean[col]):
        train_clean[col] = train_clean[col].fillna(train_clean[col].median())
    else:
        train_clean[col] = train_clean[col].fillna("Unknown")
print("Total missing values after cleaning:", train_clean.isnull().sum().sum())



# Remove duplicate rows
print("Shape before removing duplicates:", train_clean.shape)
print("Number of duplicate rows:", train_clean.duplicated().sum())
train_clean = train_clean.drop_duplicates()
print("Shape after removing duplicates:", train_clean.shape)

# Analyse data types and feature categories
print(train_clean.dtypes.value_counts())
numeric_cols = train_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = train_clean.select_dtypes(include=["object"]).columns.tolist()
if target in categorical_cols:
    categorical_cols.remove(target)
print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# Target variable distribution
print(train_clean[target].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_clean[target].value_counts().plot(kind="bar", ax=axes[0])
axes[0].set_title("Incident Grade Distribution")
axes[0].set_xlabel("Incident Grade")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

incident_counts = train_clean[target].value_counts()
axes[1].pie(incident_counts.values, labels=incident_counts.index, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Incident Grade Distribution")

plt.tight_layout()
#plt.show()

# Categorical cardinality analysis
cardinality = pd.DataFrame({
    "Column": categorical_cols,
    "Unique_Values": [train_clean[col].nunique() for col in categorical_cols],
    "Unique_Percentage": [(train_clean[col].nunique() / len(train_clean)) * 100 for col in categorical_cols]
}).sort_values("Unique_Values", ascending=False)

print(cardinality.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(data=cardinality.head(15), x="Unique_Values", y="Column")
plt.title("Top 10 High-Cardinality Categorical Features")
plt.xlabel("Number of Unique Values")
plt.ylabel("Column")
#plt.show()

# Numeric feature correlation heatmap
numeric_cols = train_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(numeric_cols) > 1:
    corr_matrix = train_clean[numeric_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric Features")
   # plt.show()
else:
    print("Not enough numeric columns for correlation analysis.")


# IncidentGrade distribution by EntityType
col = "EntityType"
if col in train_clean.columns:
    top_values = train_clean[col].value_counts().head(10).index
    temp_df = train_clean[train_clean[col].isin(top_values)]
    percentage_table = pd.crosstab(temp_df[col], temp_df[target], normalize="index") * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(percentage_table, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("IncidentGrade Percentage by EntityType")
    plt.xlabel("IncidentGrade")
    plt.ylabel("EntityType")
    #plt.show()

# Temporal feature engineering
if "Timestamp" in train_clean.columns:
    train_clean["Timestamp"] = pd.to_datetime(train_clean["Timestamp"], errors="coerce")
    train_clean["Hour"] = train_clean["Timestamp"].dt.hour
    train_clean["DayOfWeek"] = train_clean["Timestamp"].dt.dayofweek
    train_clean["Month"] = train_clean["Timestamp"].dt.month
    train_clean["IsWeekend"] = train_clean["DayOfWeek"].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)
    print(train_clean[["Timestamp", "Hour", "DayOfWeek", "Month", "IsWeekend"]].head())
else:
    print("Timestamp column not found.")


# Temporal pattern visualisations
if all(col in train_clean.columns for col in ["Hour", "DayOfWeek", "Month", "IsWeekend"]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    hour_counts = train_clean["Hour"].value_counts().sort_index()
    axes[0, 0].bar(hour_counts.index, hour_counts.values)
    axes[0, 0].set_title("Incidents by Hour of Day")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(axis="y", alpha=0.3)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = train_clean["DayOfWeek"].value_counts().sort_index()
    axes[0, 1].bar(dow_counts.index, dow_counts.values)
    axes[0, 1].set_title("Incidents by Day of Week")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(day_names)
    axes[0, 1].grid(axis="y", alpha=0.3)

    weekend_counts = train_clean["IsWeekend"].value_counts()
    axes[1, 0].bar(["Weekday", "Weekend"], [weekend_counts.get(0, 0), weekend_counts.get(1, 0)])
    axes[1, 0].set_title("Weekday vs Weekend Incidents")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(axis="y", alpha=0.3)

    month_counts = train_clean["Month"].value_counts().sort_index()
    axes[1, 1].plot(month_counts.index, month_counts.values, marker="o", linewidth=2)
    axes[1, 1].set_title("Incidents by Month")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()


# IncidentGrade by weekday vs weekend
if "IsWeekend" in train_clean.columns:
    weekend_grade = pd.crosstab(train_clean["IsWeekend"], train_clean[target], normalize="index") * 100
    weekend_grade.index = ["Weekday", "Weekend"]
    print(weekend_grade.round(2))
    weekend_grade.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("IncidentGrade Distribution: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.legend(title="IncidentGrade")
    plt.tight_layout()
    #plt.show()


# IncidentGrade distribution by hour
if "Hour" in train_clean.columns:
    hour_grade = pd.crosstab(train_clean["Hour"], train_clean[target], normalize="index") * 100
    hour_grade.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("IncidentGrade Distribution by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.legend(title="IncidentGrade")
    plt.tight_layout()
    #plt.show()


# Incident trend over time
if "Timestamp" in train_clean.columns:
    trend_data = train_clean.dropna(subset=["Timestamp"]).copy()
    incident_trend = trend_data.groupby(trend_data["Timestamp"].dt.date).size()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=incident_trend.index, y=incident_trend.values)
    plt.title("Incident Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()
else:
    print("Timestamp column not found.")


# Drop Timestamp after feature extraction
model_df = train_clean.copy()
if "Timestamp" in model_df.columns:
    model_df = model_df.drop(columns=["Timestamp"])
print("Model dataframe shape:", model_df.shape)
print(model_df.head())



# Drop ID-like and high-cardinality columns
drop_cols = [
    "Id", "OrgId", "IncidentId", "AlertId", "DeviceId", "Sha256",
    "IpAddress", "Url", "AccountSid", "AccountUpn", "AccountObjectId",
    "AccountName", "DeviceName", "NetworkMessageId", "EmailClusterId",
    "ApplicationId", "OAuthApplicationId", "ResourceIdName",
    "RegistryKey", "RegistryValueName", "RegistryValueData"
]
drop_cols = [col for col in drop_cols if col in model_df.columns]
model_df = model_df.drop(columns=drop_cols)
print("Dropped columns:", drop_cols)
print("Shape after dropping:", model_df.shape)


# Split features and target
X = model_df.drop(columns=[target])
y = model_df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts())

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print("Target encoding:")
for label, number in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)):
    print(label, "=", number)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train = X_train.copy()
X_val = X_val.copy()
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)


# Encode categorical features using OrdinalEncoder
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
print("Categorical features:", categorical_features)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
if len(categorical_features) > 0:
    X_train[categorical_features] = encoder.fit_transform(X_train[categorical_features])
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
print("ROC_AUC:", roc_auc_score(y_val, y_prod_baseline , multi_class="ovr"))
print("Baseline Macro F1:", f1_score(y_val, y_pred_baseline, average="macro"))


# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
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
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
#plt.show()


# Random Forest feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

top_features = feature_importance.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 15 Important Features - Random Forest")
plt.gca().invert_yaxis()
#plt.show()

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
    tree_method="hist"
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
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens",
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
#plt.show()


# Model comparison
model_results = pd.DataFrame({
    "Model": ["Baseline", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_val, y_pred_baseline),
        accuracy_score(y_val, y_pred_rf),
        accuracy_score(y_val, y_pred_xgb)
    ],
    "ROC_AUC": [
        roc_auc_score(y_val, y_prod_baseline, multi_class="ovr"),
        roc_auc_score(y_val, y_prod_rf, multi_class="ovr"),
        roc_auc_score(y_val, y_prod_xgb, multi_class="ovr")
    ],
    "Macro F1 Score": [
        f1_score(y_val, y_pred_baseline, average="macro"),
        f1_score(y_val, y_pred_rf, average="macro"),
        f1_score(y_val, y_pred_xgb, average="macro")
    ]
})
print(model_results)

plt.figure(figsize=(8, 5))
sns.barplot(data=model_results, x="Model", y="Macro F1 Score")
plt.title("Model Comparison Based on Macro F1 Score")
plt.xlabel("Model")
plt.ylabel("Macro F1 Score")
#plt.show()

### Saving Artifacts ####
## Saving the best result model (Random Forest).

os.makedirs (BASE_DIR / "model_store", exist_ok=True)
joblib.dump(rf_model, BASE_DIR / "model_store" / "rf_model.pkl")
joblib.dump(encoder, BASE_DIR / "model_store" / "feature_encoder.pkl")
joblib.dump(target_encoder, BASE_DIR / "model_store" / "target_encoder.pkl")

## Save future schema so the API knows what to expect.
with open(BASE_DIR / "model_store" / "feature_columns.json", "w") as f:
    json.dump(X_train.columns.tolist(), f)
