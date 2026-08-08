import pytest
import pandas as pd
import numpy as np
from src.data.validate import validate_data

def test_validate_data_drops_outliers():
    """Test that validate_data removes impossible trip durations."""
    # Create a dummy dataset with one extreme outlier (trip duration = 100 hours)
    data = {
        "pickup_datetime": pd.to_datetime(["2020-01-01 10:00:00", "2020-01-01 11:00:00"]),
        "dropoff_datetime": pd.to_datetime(["2020-01-01 10:10:00", "2020-01-05 11:00:00"]),
        "trip_distance": [1.0, 1.0],
        "passenger_count": [1, 1],
        "trip_duration": [600, 360000], # 10 mins vs ~100 hours
        "pickup_longitude": [-73.9, -73.9],
        "pickup_latitude": [40.7, 40.7],
        "dropoff_longitude": [-73.8, -73.8],
        "dropoff_latitude": [40.8, 40.8],
        "store_and_fwd_flag": [0, 0]
    }
    df = pd.DataFrame(data)
    df_clean, report = validate_data(df)
    
    # Should drop the row with 360,000 seconds
    assert len(df_clean) == 1
    assert "outliers" in report
