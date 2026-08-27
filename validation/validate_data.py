"""Validate REP's raw NYC security-incident data before feature engineering.

The validation gate intentionally runs before training transformations. If the
input schema or critical value checks fail, the process exits with a non-zero
status and downstream model code must not continue.

Usage from the repository root:
    python validation/validate_data.py
    python validation/validate_data.py --data data/NYC_Train.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "NYC_Train_raw.csv"

# These are the columns consumed by the current training workflow. Identifier
# columns are included because their presence is useful for detecting a changed
# source export, even though some are dropped later during feature engineering.
REQUIRED_COLUMNS = [
    "Id",
    "OrgId",
    "IncidentId",
    "AlertId",
    "Timestamp",
    "DetectorId",
    "AlertTitle",
    "Category",
    "MitreTechniques",
    "IncidentGrade",
    "ActionGrouped",
    "ActionGranular",
    "EntityType",
    "EvidenceRole",
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
    "RegistryKey",
    "RegistryValueName",
    "RegistryValueData",
    "ApplicationId",
    "ApplicationName",
    "OAuthApplicationId",
    "ThreatFamily",
    "FileName",
    "FolderPath",
    "ResourceIdName",
    "ResourceType",
    "Roles",
    "OSFamily",
    "OSVersion",
    "AntispamDirection",
    "SuspicionLevel",
    "LastVerdict",
    "CountryCode",
    "State",
    "City",
]

NUMERIC_COLUMNS = [
    "Id",
    "OrgId",
    "IncidentId",
    "AlertId",
    "DetectorId",
    "AlertTitle",
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
    "RegistryKey",
    "RegistryValueName",
    "RegistryValueData",
    "ApplicationId",
    "OAuthApplicationId",
    "FileName",
    "FolderPath",
    "ResourceIdName",
    "OSFamily",
    "OSVersion",
    "CountryCode",
    "State",
    "City",
]

CATEGORICAL_COLUMNS = [
    "Category",
    "MitreTechniques",
    "IncidentGrade",
    "ActionGrouped",
    "ActionGranular",
    "EntityType",
    "EvidenceRole",
    "ThreatFamily",
    "ResourceType",
    "Roles",
    "AntispamDirection",
    "SuspicionLevel",
    "LastVerdict",
]

ALLOWED_GRADES = ["BenignPositive", "FalsePositive", "TruePositive"]


def build_schema() -> DataFrameSchema:
    """Build the Pandera schema for the raw REP training export."""
    columns = {
        column: Column(float if column == "EmailClusterId" else int, nullable=True)
        for column in NUMERIC_COLUMNS
    }
    columns.update({
        column: Column(object, nullable=True) for column in CATEGORICAL_COLUMNS
    })
    columns["Timestamp"] = Column(object, nullable=False)

    # The training script removes rows with a missing target. Therefore a raw
    # export may contain missing IncidentGrade values, but non-null values must
    # be one of the known labels.
    columns["IncidentGrade"] = Column(
        object,
        nullable=True,
        checks=Check.isin(ALLOWED_GRADES),
    )

    return DataFrameSchema(
        columns=columns,
        checks=[
            Check(lambda frame: len(frame) > 0, error="dataset must not be empty"),
        ],
        strict=False,
        coerce=False,
    )


SCHEMA = build_schema()


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return ``df`` if it passes the REP data gate.

    Schema violations raise ``pandera.errors.SchemaErrors``. File-level
    semantic violations raise ``ValueError``. Both are intentionally allowed
    to stop the pipeline.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate() expects a pandas DataFrame")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    validated = SCHEMA.validate(df, lazy=True)

    timestamp_values = pd.to_datetime(
        validated["Timestamp"], errors="coerce", utc=True
    )
    if timestamp_values.isna().any():
        invalid_count = int(timestamp_values.isna().sum())
        raise ValueError(
            f"Timestamp contains {invalid_count} invalid or missing value(s)"
        )

    # These columns are used as IDs/features and should not contain negative
    # values. Missing optional numeric values are allowed and handled later.
    non_negative_columns = [
        column for column in NUMERIC_COLUMNS if column != "EmailClusterId"
    ]
    negative_counts = {
        column: int((validated[column].dropna() < 0).sum())
        for column in non_negative_columns
    }
    negative_counts = {
        column: count for column, count in negative_counts.items() if count
    }
    if negative_counts:
        details = ", ".join(
            f"{column}={count}" for column, count in negative_counts.items()
        )
        raise ValueError(f"Negative values found in non-negative columns: {details}")

    duplicate_count = int(validated.duplicated().sum())
    if duplicate_count:
        print(
            f"WARNING: found {duplicate_count} duplicate row(s); "
            "training will remove them."
        )

    missing_target_count = int(validated["IncidentGrade"].isna().sum())
    if missing_target_count:
        print(
            f"WARNING: found {missing_target_count} row(s) without IncidentGrade; "
            "training will exclude them."
        )

    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the raw REP NYC security-incident dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV file to validate (default: {DEFAULT_DATA_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else BASE_DIR / args.data

    if not data_path.exists():
        print(f"VALIDATION FAILED: data file not found: {data_path}", file=sys.stderr)
        return 1

    try:
        print(f"Loading raw data from: {data_path}")
        data = pd.read_csv(data_path, low_memory=False)
        validated = validate(data)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors, TypeError, ValueError) as error:
        print("VALIDATION FAILED. Pipeline halted.", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    except Exception as error:
        print(f"VALIDATION FAILED while reading data: {error}", file=sys.stderr)
        return 1

    print(
        f"VALIDATION PASSED: {len(validated)} rows, "
        f"{len(validated.columns)} columns"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
