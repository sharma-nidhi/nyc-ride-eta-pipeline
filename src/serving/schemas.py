# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

from pydantic import BaseModel, Field

from src.contract import (
    NYC_LAT_MIN, NYC_LAT_MAX,
    NYC_LON_MIN, NYC_LON_MAX,
    PASSENGER_MIN, PASSENGER_MAX,
    VENDOR_MIN, VENDOR_MAX,
    STORE_AND_FWD_PATTERN,
)


class PredictionRequest(BaseModel):
    """Raw input fields a client sends — pipeline transforms them into model features."""

    pickup_datetime: str = Field(
        ...,
        description="ISO-8601 timestamp of pickup (e.g. '2016-01-01T10:30:00')",
        examples=["2016-05-15T14:30:00"],
    )
    passenger_count: int = Field(
        ...,
        ge=PASSENGER_MIN,
        le=PASSENGER_MAX,
        description="Number of passengers (1-6)",
        examples=[2],
    )
    pickup_latitude: float = Field(
        ...,
        ge=NYC_LAT_MIN,
        le=NYC_LAT_MAX,
        description="Pickup latitude (NYC range ~40.5-40.9)",
        examples=[40.748817],
    )
    pickup_longitude: float = Field(
        ...,
        ge=NYC_LON_MIN,
        le=NYC_LON_MAX,
        description="Pickup longitude (NYC range ~-74.0 to -73.7)",
        examples=[-73.985428],
    )
    dropoff_latitude: float = Field(
        ...,
        ge=NYC_LAT_MIN,
        le=NYC_LAT_MAX,
        description="Dropoff latitude",
        examples=[40.742563],
    )
    dropoff_longitude: float = Field(
        ...,
        ge=NYC_LON_MIN,
        le=NYC_LON_MAX,
        description="Dropoff longitude",
        examples=[-73.98748],
    )
    vendor_id: int = Field(
        ...,
        ge=VENDOR_MIN,
        le=VENDOR_MAX,
        description="Vendor ID (1 or 2)",
        examples=[1],
    )
    store_and_fwd_flag: str = Field(
        ...,
        pattern=STORE_AND_FWD_PATTERN,
        description="Store-and-forward flag ('N' or 'Y')",
        examples=["N"],
    )


class PredictionResponse(BaseModel):
    """Single prediction result."""

    eta_seconds: float = Field(
        ...,
        description="Predicted trip duration in seconds",
        examples=[900.5],
    )


class BatchPredictionRequest(BaseModel):
    """Multiple prediction requests."""

    requests: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prediction requests (max 100 per batch)",
    )


class BatchPredictionResponse(BaseModel):
    """Multiple prediction results."""

    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction responses",
    )


class ModelInfoResponse(BaseModel):
    """Model metadata endpoint response."""

    model_type: str = Field(..., description="Champion model type")
    run_id: str = Field(..., description="MLflow run ID for traceability")
    metrics: dict = Field(..., description="Validation metrics (mae, rmse, r2)")
    feature_count: str = Field(..., description="Number of model features")
    schema_version: str = Field(..., description="Feature schema version")
