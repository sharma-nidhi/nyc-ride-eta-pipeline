"""
Prediction logging to SQLite (Week 4 / M5).

PLACEHOLDER — persist every prediction (timestamp, inputs, predicted ETA, model_version)
so drift and error can be analysed later. Wire init()/log() into serving/api.py.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path("monitoring/predictions.db")


def init() -> None:
    raise NotImplementedError("TODO Week 4: CREATE TABLE predictions(...)")


def log(inputs: dict, result: dict) -> None:
    raise NotImplementedError("TODO Week 4: INSERT one prediction row")
