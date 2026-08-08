# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import pandas as pd
import pathlib
import logging
import joblib
from src.data.ingest import load_raw
from src.data.validate import validate_data
from src.features.feature_pipeline import build_pipeline, save_pipeline

logger = logging.getLogger(__name__)

# Paths
PROCESSED_X = pathlib.Path("data/processed/X_train.parquet")
PROCESSED_Y = pathlib.Path("data/processed/y_train.parquet")

def run_preprocessing_pipeline(sample_mode: bool = False):
    """
    Full end-to-end preprocessing: Ingest -> Validate -> Transform -> Save.
    """
    # 1. Ingest
    df_raw = load_raw(sample_mode=sample_mode)
    
    # 2. Validate
    logger.info("Running data validation...")
    df_clean, report = validate_data(df_raw)
    
    # 3. Separate Target and Features
    # Target: trip_duration (in seconds)
    y = df_clean["trip_duration"].copy()
    
    # Feature set (following the a_event contract in feature_meta.json)
    X = df_clean.drop(columns=["trip_duration"])
    
    # 4. Apply Feature Pipeline
    logger.info("Applying feature engineering pipeline...")
    pipeline = build_pipeline()
    
    # We 'fit' on the data and 'transform' it
    X_processed = pipeline.fit_transform(X)
    
    # Convert processed numpy array back to DataFrame for easier saving
    # Column order is determined by the ColumnTransformer: 
    # [num_cols] + [cat_cols] + [passthrough_cols]
    from src.features.feature_pipeline import NUMERIC_COLS, CATEGORICAL_COLS, PASSTHROUGH_COLS
    feature_names = NUMERIC_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS
    
    X_final = pd.DataFrame(X_processed, columns=feature_names, index=y.index)
    y_final = y.astype("float32")
    
    # 5. Save and Version
    PROCESSED_X.parent.mkdir(parents=True, exist_ok=True)
    
    # We use Parquet instead of CSV for processed data (preserves types, much smaller/faster)
    X_final.to_parquet(PROCESSED_X)
    y_final.to_frame(name="target").to_parquet(PROCESSED_Y)
    
    # Save the pipeline artifact for use in the API later
    save_pipeline(pipeline)
    
    logger.info("Preprocessing complete.")
    logger.info(f"Saved processed features to {PROCESSED_X}")
    logger.info(f"Saved processed target to {PROCESSED_Y}")
    logger.info(f"Saved pipeline artifact to models/feature_pipeline.pkl")
    
    return X_final, y_final

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        # For first run, use sample_mode=True to verify it works quickly
        # Once verified, change to False to process the full 1.4M records
        X, y = run_preprocessing_pipeline(sample_mode=True)
        print("\n--- Final Processed Data Summary ---")
        print(f"Features shape: {X.shape}")
        print(f"Target shape:   {y.shape}")
        print("\nFirst 5 rows of processed features:\n", X.head())
        
    except Exception as e:
        logger.error("Preprocessing pipeline failed: %s", e, exc_info=True)