# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Preprocessing Pipeline
=======================
End-to-end data preparation: ingests raw CSV, runs schema validation,
engineers features, splits chronologically (80/20), and exports
`X_train.parquet`, `y_train.parquet`, and `feature_pipeline.pkl`.
Also writes a feature registry JSON for serving-side validation."""

import pandas as pd
import pathlib
import logging
import joblib
import json
from src.data.ingest import load_raw
from src.data.validate import validate_data
from src.features.feature_pipeline import build_pipeline, save_pipeline

logger = logging.getLogger(__name__)

# Paths
PROCESSED_X = pathlib.Path("data/processed/X_train.parquet")
PROCESSED_Y = pathlib.Path("data/processed/y_train.parquet")
FEATURE_REGISTRY_PATH = pathlib.Path("data/contracts/feature_registry.json")

def run_preprocessing_pipeline(sample_mode: bool = False, end_month: int | None = None):
    """
    Full end-to-end preprocessing: Ingest -> Validate -> Transform -> Save.
    
    Args:
        sample_mode: Load only a sample of rows for quick testing.
        end_month: Filter data to pickup dates before this month (1-12).
                   None = no filter (full dataset).
    """
    # 1. Ingest
    df_raw = load_raw(sample_mode=sample_mode, end_month=end_month)
    
    # 2. Validate
    logger.info("Running data validation...")
    df_clean, report = validate_data(df_raw)
    
    # 3. Separate Target and Features
    y = df_clean["trip_duration"].copy()
    X = df_clean.drop(columns=["trip_duration"])
    
    # 4. Apply Feature Pipeline
    logger.info("Applying feature engineering pipeline...")
    pipeline = build_pipeline()
    X_processed = pipeline.fit_transform(X)
    
    # Import feature lists from the single source of truth
    from src.features.feature_config import NUMERIC_COLS, CATEGORICAL_COLS, PASSTHROUGH_COLS
    feature_names = NUMERIC_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS
    
    X_final = pd.DataFrame(X_processed, columns=feature_names, index=y.index)
    y_final = y.astype("float32")
    
    # 5. Save
    PROCESSED_X.parent.mkdir(parents=True, exist_ok=True)
    X_final.to_parquet(PROCESSED_X)
    y_final.to_frame(name="target").to_parquet(PROCESSED_Y)
    save_pipeline(pipeline)

    # 6. Auto-generate Feature Registry (Lightweight Feature Store contract)
    from src.features.feature_config import get_feature_schema
    
    schema = get_feature_schema()
    schema["dvc_slice"] = f"end_month={end_month}" if end_month else "full"
    schema["row_count"] = len(X_final)
    schema["generated_at"] = pd.Timestamp.now().isoformat()

    FEATURE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_REGISTRY_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Feature registry synchronized at %s", FEATURE_REGISTRY_PATH)
    
    logger.info("Preprocessing complete.")
    logger.info(f"Saved processed features to {PROCESSED_X} ({len(X_final)} rows)")
    logger.info(f"Saved processed target to {PROCESSED_Y}")
    
    return X_final, y_final

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Preprocessing pipeline with optional data slicing")
    parser.add_argument("--end-month", type=int, default=None, 
                        help="Filter data to months 1-N (e.g. --end-month 1 for Jan only)")
    parser.add_argument("--sample", action="store_true", help="Load sample mode for quick testing")
    args = parser.parse_args()
    
    try:
        X, y = run_preprocessing_pipeline(sample_mode=args.sample, end_month=args.end_month)
        print("\n--- Final Processed Data Summary ---")
        print(f"Features shape: {X.shape}")
        print(f"Target shape:   {y.shape}")
        
    except Exception as e:
        logger.error("Preprocessing pipeline failed: %s", e, exc_info=True)