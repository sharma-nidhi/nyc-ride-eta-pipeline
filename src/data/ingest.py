# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Data Ingestion
===============
Loads raw NYC taxi trip CSV data with memory-efficient dtypes and optional
sample-mode for quick experimentation. Outputs a raw, unmodified DataFrame."""

import pandas as pd
import pathlib
import logging

logger = logging.getLogger(__name__)

RAW_PATH = pathlib.Path("data/raw/NYC.csv")

# Explicit dtypes cut memory usage by ~60% vs default float64/int64
DTYPES = {
    "vendor_id":          "int8",
    "passenger_count":    "int8",
    "pickup_longitude":   "float32",
    "pickup_latitude":    "float32",
    "dropoff_longitude":  "float32",
    "dropoff_latitude":   "float32",
    "store_and_fwd_flag": "category",
    "trip_duration":      "int32",
}

# Only load columns that are either at_event features or needed to derive the target
USECOLS = [
    "vendor_id",
    "pickup_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration",       # target variable (seconds)
]


def load_raw(sample_mode: bool = False, sample_size: int = 100_000, end_month: int | None = None) -> pd.DataFrame:
    """Load NYC Taxi raw CSV with memory-optimised dtypes.

    Args:
        sample_mode: If True, loads only `sample_size` rows. Use during development.
        sample_size: Number of rows to load in sample mode.
        end_month: Filter rows to pickup_datetime <= end of given month (1-12).
                   None = no filter (full dataset).

    Returns:
        Raw DataFrame with correct column types.
    """
    nrows = sample_size if sample_mode else None
    mode_label = f"sample ({sample_size:,} rows)" if sample_mode else "full dataset"
    logger.info("Loading %s from %s", mode_label, RAW_PATH)

    df = pd.read_csv(
        RAW_PATH,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["pickup_datetime"],
        nrows=nrows,
    )

    # Sort chronologically — required for time-based train/test split (M2 rule)
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    # Time-based slicing for DVC dataset versioning
    if end_month is not None:
        cutoff = pd.Timestamp(year=2016, month=end_month + 1, day=1)
        before = len(df)
        df = df[df["pickup_datetime"] < cutoff].reset_index(drop=True)
        logger.info("Sliced to month %d: %d → %d rows", end_month, before, len(df))

    logger.info(
        "Loaded %d rows | Memory: %.1f MB",
        len(df),
        df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Quick schema check without loading the full file
    peek = pd.read_csv(RAW_PATH, nrows=2)
    print("Columns found:", peek.columns.tolist())
    print()

    # Load a sample to verify before committing to the full dataset
    df = load_raw(sample_mode=True, sample_size=5_000)
    print(df.dtypes)
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB for 5k rows")
    print(f"Estimated full dataset: {df.memory_usage(deep=True).sum() / 1e6 * (1_458_644 / 5_000):.0f} MB")