
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path("monitoring/predictions.db")


def init():
    """Create the predictions table once (safe to call on every startup)."""
    DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp          TEXT,
            pickup_datetime    TEXT,
            pickup_longitude   REAL,
            pickup_latitude    REAL,
            dropoff_longitude  REAL,
            dropoff_latitude   REAL,
            passenger_count    INTEGER,
            vendor_id          INTEGER,
            store_and_fwd_flag TEXT,
            haversine_km       REAL,
            eta_seconds        REAL,
            model_version      TEXT
        )
    """)
    conn.commit()
    conn.close()


def log(inputs: dict, features: dict, eta_seconds: float, model_version: str):
    """Persist one prediction (raw inputs + a key feature + the output)."""
    conn = sqlite3.connect(DB)
    conn.execute(
        """INSERT INTO predictions (
            timestamp, pickup_datetime, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, passenger_count, vendor_id,
            store_and_fwd_flag, haversine_km, eta_seconds, model_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            inputs["pickup_datetime"], inputs["pickup_longitude"], inputs["pickup_latitude"],
            inputs["dropoff_longitude"], inputs["dropoff_latitude"], inputs["passenger_count"],
            inputs["vendor_id"], inputs["store_and_fwd_flag"],
            features["haversine_km"], eta_seconds, model_version,
        ),
    )
    conn.commit()
    conn.close()