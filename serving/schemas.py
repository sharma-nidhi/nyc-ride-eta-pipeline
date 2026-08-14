"""
Pydantic request/response contracts for the ETA API (Week 3 / M4).

PLACEHOLDER — tighten field constraints to match validation ranges in config.yaml.
"""
from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    """Input contract — one trip. FastAPI auto-validates and returns 422 on bad input."""
    pickup_datetime: str = Field(..., description="ISO datetime, e.g. 2016-03-14T17:24:00")
    pickup_longitude: float = Field(..., ge=-74.3, le=-73.7)
    pickup_latitude: float = Field(..., ge=40.5, le=41.0)
    dropoff_longitude: float = Field(..., ge=-74.3, le=-73.7)
    dropoff_latitude: float = Field(..., ge=40.5, le=41.0)
    passenger_count: int = Field(..., ge=1, le=6)
    # TODO: add vendor_id / store_and_fwd_flag / weather if used as features


class ETAResponse(BaseModel):
    """Output contract."""
    eta_seconds: float
    eta_minutes: float
    model_version: str
