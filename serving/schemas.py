
from typing import Literal
from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    """Input contract — one trip. FastAPI auto-validates; bad input -> HTTP 422."""
    pickup_datetime: str = Field(..., description="ISO datetime, e.g. 2016-03-14T17:24:00")
    pickup_longitude: float = Field(..., ge=-74.3, le=-73.7)
    pickup_latitude: float = Field(..., ge=40.5, le=41.0)
    dropoff_longitude: float = Field(..., ge=-74.3, le=-73.7)
    dropoff_latitude: float = Field(..., ge=40.5, le=41.0)
    passenger_count: int = Field(..., ge=1, le=6)
    vendor_id: int = Field(1, ge=1, le=2, description="taxi vendor (default 1)")
    store_and_fwd_flag: Literal["Y", "N"] = Field("N", description="store-and-forward flag")


class ETAResponse(BaseModel):
    """Output contract."""
    eta_seconds: float
    eta_minutes: float
    model_version: str