
import os
import sys
import pandas as pd

RAW_PATH = "data/raw/train.csv"
PROCESSED_PATH = "data/processed/train_clean.parquet"
REPORT_PATH = "data/processed/validation_report.txt"

REQUIRED_COLUMNS = [
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime",
    "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag",
    "trip_duration",
]

# Quality thresholds
MIN_DURATION_S, MAX_DURATION_S = 30, 7200      # 30 sec .. 2 hours
MIN_LAT, MAX_LAT = 40.5, 41.0                  # NYC latitude bounds
MIN_LON, MAX_LON = -74.3, -73.7                # NYC longitude bounds
MIN_PASSENGERS, MAX_PASSENGERS = 1, 6


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def check_schema(df: pd.DataFrame) -> None:
    """Check that the DataFrame has all the required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  FAIL: missing required columns: {missing}")
        sys.exit(1)
    print("  PASS: all required columns present")


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the datetime text columns into real datetime objects."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    print("  PASS: pickup/dropoff parsed to datetime")
    return df


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible / out-of-range rows and report how many go per rule."""
    n_start = len(df)
    report = []

    def keep(mask, reason):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        report.append((reason, before - len(df)))

    keep(df["dropoff_datetime"] > df["pickup_datetime"], "dropoff not after pickup")
    keep(df["trip_duration"].between(MIN_DURATION_S, MAX_DURATION_S),
         f"trip_duration outside [{MIN_DURATION_S}, {MAX_DURATION_S}] s")
    keep(df["pickup_latitude"].between(MIN_LAT, MAX_LAT), "pickup_latitude outside NYC")
    keep(df["pickup_longitude"].between(MIN_LON, MAX_LON), "pickup_longitude outside NYC")
    keep(df["dropoff_latitude"].between(MIN_LAT, MAX_LAT), "dropoff_latitude outside NYC")
    keep(df["dropoff_longitude"].between(MIN_LON, MAX_LON), "dropoff_longitude outside NYC")
    keep(df["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS),
         f"passenger_count outside [{MIN_PASSENGERS}, {MAX_PASSENGERS}]")
    keep(df["store_and_fwd_flag"].isin(["Y", "N"]), "store_and_fwd_flag not Y/N")

    lines = ["Quality filter report:"]
    for reason, removed in report:
        lines.append(f"  dropped {removed:>8,}  ({reason})")
    n_kept = len(df)
    lines.append("  ---")
    lines.append(f"  kept {n_kept:,} of {n_start:,} rows ({n_kept / n_start * 100:.2f}%)")
    for line in lines:
        print(line)
    return df, lines


def save_clean(df: pd.DataFrame, report_lines: list) -> None:
    """Persist the cleaned dataset (Parquet) and the validation report (text)."""
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved cleaned data -> {PROCESSED_PATH}  ({len(df):,} rows)")
    print(f"Saved report       -> {REPORT_PATH}")


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    check_schema(df)
    df = parse_datetimes(df)
    df, report_lines = apply_quality_filters(df)
    save_clean(df, report_lines)
    print("\nValidation complete — cleaned dataset ready for feature engineering.")