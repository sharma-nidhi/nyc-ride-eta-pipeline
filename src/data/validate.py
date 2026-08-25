# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import logging
import json
import pathlib
from typing import Tuple, Dict, Any

from src.contract import (
    NYC_LAT_MIN, NYC_LAT_MAX,
    NYC_LON_MIN, NYC_LON_MAX,
    PASSENGER_MIN, PASSENGER_MAX,
    DURATION_MIN, DURATION_MAX,
    SPEED_MAX,
)

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates the taxi dataset for nulls, ranges, and physical impossibility.
    
    Returns:
        cleaned_df: DataFrame with rows failing hard rules removed.
        report: Dictionary containing failure counts and percentages.
    """
    initial_count = len(df)
    report = {"total_rows": initial_count, "failures": {}}
    
    # 1. Null Checks
    # Rows must have these core fields to be useful for training
    critical_cols = [
        "pickup_datetime", "pickup_longitude", "pickup_latitude", 
        "dropoff_longitude", "dropoff_latitude", "trip_duration"
    ]
    null_mask = df[critical_cols].isnull().any(axis=1)
    report["failures"]["nulls"] = int(null_mask.sum())
    
    # 2. Range Checks
    # Passenger count must be physically possible
    pass_mask = (df["passenger_count"] < PASSENGER_MIN) | (df["passenger_count"] > PASSENGER_MAX)
    report["failures"]["invalid_passenger_count"] = int(pass_mask.sum())

    # Bounding Box check: coordinates must be within the defined NYC range
    lat_mask = (df["pickup_latitude"] < NYC_LAT_MIN) | (df["pickup_latitude"] > NYC_LAT_MAX) | \
               (df["dropoff_latitude"] < NYC_LAT_MIN) | (df["dropoff_latitude"] > NYC_LAT_MAX)
    lon_mask = (df["pickup_longitude"] < NYC_LON_MIN) | (df["pickup_longitude"] > NYC_LON_MAX) | \
               (df["dropoff_longitude"] < NYC_LON_MIN) | (df["dropoff_longitude"] > NYC_LON_MAX)
    report["failures"]["out_of_bounds_coords"] = int((lat_mask | lon_mask).sum())
    
    # 3. Logic Checks
    # Trip duration must be positive, at least 60s (real ride), and < 4 hours
    duration_mask = (df["trip_duration"] <= 60) | (df["trip_duration"] > 14400)  # 4 hours in seconds
    report["failures"]["invalid_duration"] = int(duration_mask.sum())

    # Zero-distance trips: pickup and dropoff coordinates are identical
    # These will produce haversine_dist=0 and are unusable for ETA prediction
    zero_dist_mask = (
        (df["pickup_latitude"] == df["dropoff_latitude"]) &
        (df["pickup_longitude"] == df["dropoff_longitude"])
    )
    report["failures"]["zero_distance_trip"] = int(zero_dist_mask.sum())

    # Speed check: compute haversine distance to detect physically impossible trips
    # Any trip implying speed > 150 km/h in NYC is a sensor/data error
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_km = 6371 * 2 * np.arcsin(np.sqrt(a))
    speed_kmh = dist_km / (df["trip_duration"] / 3600)
    speed_mask = speed_kmh > 150
    report["failures"]["impossible_speed"] = int(speed_mask.sum())

    # Combined failure mask: any row failing any rule is marked for removal
    combined_mask = null_mask | pass_mask | lat_mask | lon_mask | duration_mask | zero_dist_mask | speed_mask
    
    # Filter the dataframe to keep only valid records
    cleaned_df = df[~combined_mask].copy()
    
    # Calculate final loss and percentage
    dropped_count = initial_count - len(cleaned_df)
    drop_pct = (dropped_count / initial_count) * 100
    report["total_dropped"] = dropped_count
    report["drop_percentage"] = drop_pct
    
    logger.info("Validation complete. Dropped %.2f%% (%d rows)", drop_pct, dropped_count)
    
    # Hard-fail threshold: If > 10% of data is bad, the source is likely corrupted
    if drop_pct > 10.0:
        raise ValueError(f"Data quality critical failure: {drop_pct:.2f}% of records failed validation.")
        
    return cleaned_df, report

if __name__ == "__main__":
    import logging
    import sys
    
    # Add current directory to path so we can import ingest
    sys.path.append('.')
    try:
        from src.data.ingest import load_raw
    except ImportError:
        print("Error: ensure you are running from the project root folder.")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        # 1. Ingest a sample for testing
        df_raw = load_raw(sample_mode=True, sample_size=100_000)
        
        # 2. Run the validation pipeline
        df_clean, report = validate_data(df_raw)
        
        # 3. Save the validation report for the assignment audit trail
        pathlib.Path("data").mkdir(parents=True, exist_ok=True)
        out_path = pathlib.Path("data/validation_report.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
            
        print("\n--- Validation Summary ---")
        print(json.dumps(report, indent=2))
        print(f"\nCleaned data preserved in memory. Report written to {out_path}")
        
    except Exception as e:
        print(f"\nPipeline Failure: {e}")