# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
API Integration Tests
=====================
Integration tests for the FastAPI prediction service. Covers valid
single/batch requests, invalid inputs (bad coordinates, out-of-range
passengers, missing fields), health check, and model-info endpoints."""

from fastapi.testclient import TestClient

from src.serving.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_REQUEST = {
    "pickup_datetime": "2016-05-15T14:30:00Z",
    "passenger_count": 2,
    "pickup_latitude": 40.748817,
    "pickup_longitude": -73.985428,
    "dropoff_latitude": 40.742563,
    "dropoff_longitude": -73.98748,
    "vendor_id": 1,
    "store_and_fwd_flag": "N",
}


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_model_info_returns_200(self):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] in ("LIGHTGBM", "XGBOOST", "CATBOOST", "RIDGE")
        assert "mae" in data["metrics"]
        assert "r2" in data["metrics"]


# ---------------------------------------------------------------------------
# Single Prediction — Valid
# ---------------------------------------------------------------------------

class TestSinglePrediction:
    def test_valid_prediction(self):
        resp = client.post("/predict", json=VALID_REQUEST)
        assert resp.status_code == 200
        data = resp.json()
        assert "eta_seconds" in data
        assert data["eta_seconds"] > 0

    def test_prediction_returns_reasonable_eta(self):
        """ETA should be within cleaned data range (60s–14400s) — matches validate.py."""
        resp = client.post("/predict", json=VALID_REQUEST)
        eta = resp.json()["eta_seconds"]
        assert 60 <= eta <= 14400

    def test_same_input_produces_same_eta(self):
        """Idempotency: identical requests must return identical ETAs."""
        resp1 = client.post("/predict", json=VALID_REQUEST)
        resp2 = client.post("/predict", json=VALID_REQUEST)
        assert resp1.json()["eta_seconds"] == resp2.json()["eta_seconds"]

    def test_eta_matches_training_data_bounds(self):
        """Verify ETA is at least 60 seconds (min trip duration from validate.py)."""
        resp = client.post("/predict", json=VALID_REQUEST)
        assert resp.json()["eta_seconds"] >= 60


# ---------------------------------------------------------------------------
# Batch Prediction — Valid
# ---------------------------------------------------------------------------

class TestBatchPrediction:
    def test_valid_batch(self):
        resp = client.post("/predict/batch", json={"requests": [VALID_REQUEST, VALID_REQUEST]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        for p in data["predictions"]:
            assert p["eta_seconds"] > 0

    def test_batch_max_limit(self):
        """Batch with exactly 100 requests should work."""
        resp = client.post("/predict/batch", json={"requests": [VALID_REQUEST] * 100})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 100


# ---------------------------------------------------------------------------
# Input Validation — Invalid requests (expect 422)
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_required_field(self):
        bad = VALID_REQUEST.copy()
        del bad["pickup_datetime"]
        assert client.post("/predict", json=bad).status_code == 422

    def test_invalid_passenger_count(self):
        bad = VALID_REQUEST.copy()
        bad["passenger_count"] = 10  # max is 6
        assert client.post("/predict", json=bad).status_code == 422

    def test_invalid_vendor_id(self):
        bad = VALID_REQUEST.copy()
        bad["vendor_id"] = 3  # only 1 or 2
        assert client.post("/predict", json=bad).status_code == 422

    def test_pickup_datetime_requires_timezone(self):
        bad = VALID_REQUEST.copy()
        bad["pickup_datetime"] = "2016-05-15T14:30:00"  # missing timezone suffix
        assert client.post("/predict", json=bad).status_code == 422

    def test_invalid_store_and_fwd_flag(self):
        bad = VALID_REQUEST.copy()
        bad["store_and_fwd_flag"] = "X"  # only N or Y
        assert client.post("/predict", json=bad).status_code == 422

    def test_out_of_range_latitude(self):
        bad = VALID_REQUEST.copy()
        bad["pickup_latitude"] = 90.0  # max is 50
        assert client.post("/predict", json=bad).status_code == 422

    def test_negative_longitude(self):
        bad = VALID_REQUEST.copy()
        bad["pickup_longitude"] = -90.0  # min is -82
        assert client.post("/predict", json=bad).status_code == 422

    def test_batch_exceeds_max(self):
        resp = client.post("/predict/batch", json={"requests": [VALID_REQUEST] * 101})
        assert resp.status_code == 422
