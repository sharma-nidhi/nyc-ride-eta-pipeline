"""Build and persist the model features used by the REP classifier.

This module is the feature-engineering boundary for the incident-grade model.
It mirrors the preparation steps in ``training/train_model_from_store.py``:
missing-value handling, duplicate removal, timestamp features, and removal of
identifier/high-cardinality columns.

The resulting SQLite table contains the model features plus ``IncidentGrade``
so it can be used for both training and feature inspection.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "NYC_Train_raw.csv"
FEATURE_STORE_DB = BASE_DIR / "feature_store" / "feature_store.db"
TABLE_NAME = "incident_features"
TARGET_COLUMN = "IncidentGrade"

# These are the same columns removed by train_model_from_store.py.
DROP_COLUMNS = {
    "Id",
    "OrgId",
    "IncidentId",
    "AlertId",
    "DeviceId",
    "Sha256",
    "IpAddress",
    "Url",
    "AccountSid",
    "AccountUpn",
    "AccountObjectId",
    "AccountName",
    "DeviceName",
    "NetworkMessageId",
    "EmailClusterId",
    "ApplicationId",
    "OAuthApplicationId",
    "ResourceIdName",
    "RegistryKey",
    "RegistryValueName",
    "RegistryValueData",
}

TIME_FEATURES = ("Hour", "DayOfWeek", "Month", "IsWeekend")


def load_raw(path: Path | str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw incident export."""
    return pd.read_csv(path, low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable rows and fill missing feature values."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Input data must contain {TARGET_COLUMN!r}")

    cleaned = df.dropna(subset=[TARGET_COLUMN]).drop_duplicates().copy()

    for column in cleaned.columns:
        if column == TARGET_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the time features used by the training workflow."""
    result = df.copy()
    if "Timestamp" not in result.columns:
        raise ValueError("Input data must contain 'Timestamp'")

    timestamp = pd.to_datetime(result["Timestamp"], errors="coerce", utc=True)
    if timestamp.isna().any():
        invalid_count = int(timestamp.isna().sum())
        raise ValueError(f"Timestamp contains {invalid_count} invalid value(s)")

    result["Hour"] = timestamp.dt.hour
    result["DayOfWeek"] = timestamp.dt.dayofweek
    result["Month"] = timestamp.dt.month
    result["IsWeekend"] = (result["DayOfWeek"] >= 5).astype(int)
    return result.drop(columns=["Timestamp"])


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned, model-ready feature table with its target."""
    result = add_time_features(clean_data(df))
    result = result.drop(columns=[column for column in DROP_COLUMNS if column in result])

    expected_columns = [
        column for column in result.columns if column != TARGET_COLUMN
    ]
    missing_time_features = [
        column for column in TIME_FEATURES if column not in expected_columns
    ]
    if missing_time_features:
        raise ValueError(
            "Feature construction failed; missing columns: "
            + ", ".join(missing_time_features)
        )

    # Keep the target last and preserve the source-column order used by training.
    return result[expected_columns + [TARGET_COLUMN]]


def save_to_feature_store(
    features: pd.DataFrame,
    database_path: Path | str = FEATURE_STORE_DB,
) -> None:
    """Replace the feature-store table with the newly built feature set."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        features.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)

    print(
        f"Feature store updated: {len(features)} rows, "
        f"{len(features.columns) - 1} features + target"
    )
    print(f"Table: {TABLE_NAME}")
    print(f"Features: {list(features.drop(columns=[TARGET_COLUMN]).columns)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build REP incident features")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the raw incident CSV",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=FEATURE_STORE_DB,
        help="Path to the SQLite feature store",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading raw data from: {args.data}")
    print('Building feature set...')
    features = build_feature_set(load_raw(args.data))
    print('Persisting to feature store...')
    save_to_feature_store(features, args.database)


if __name__ == "__main__":
    main()
