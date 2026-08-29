# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Simulates production traffic to test monitoring and drift detection.
Generates synthetic requests with realistic drift scenarios.

Drift Scenarios
────────────────
 normal   – Historical rows sampled from the training distribution (baseline)
 rush     – Concentrated in peak hours (7-9AM, 5-7PM), longer trip distances
 suburban – Pickups shift outside Manhattan (larger trip radii)
 holiday  – Weekend-heavy, fewer passengers, short local trips
"""
import json
import time
import random
import datetime
import numpy as np
import pandas as pd
import requests
import logging

from src.contract import (
    NYC_LAT_MIN,
    NYC_LAT_MAX,
    NYC_LON_MIN,
    NYC_LON_MAX,
    PASSENGER_MIN,
    PASSENGER_MAX,
    VENDOR_MIN,
    VENDOR_MAX,
)
from src.data.ingest import load_raw
from src.reproducibility import DEFAULT_RANDOM_SEED, seed_everything

logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/predict"

# NYC bounding boxes by zone
BOROUGH_REGIONS = {
    "Manhattan": {
        "lat": (40.748, 40.817),
        "lon": (-74.010, -73.944),
    },
    "Brooklyn": {
        "lat": (40.570, 40.700),
        "lon": (-74.020, -73.850),
    },
    "Queens": {
        "lat": (40.540, 40.800),
        "lon": (-73.960, -73.700),
    },
}

SERVICE_OPTIONS = ["uberX", "comfort", "uberXL"]
RAW_REQUIRED_COLS = [
    "pickup_datetime",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "vendor_id",
    "store_and_fwd_flag",
]


def _clamp_coords(lat: float, lon: float) -> tuple[float, float]:
    """Clamp coordinates to shared API contract bounds."""
    lat = max(NYC_LAT_MIN, min(NYC_LAT_MAX, lat))
    lon = max(NYC_LON_MIN, min(NYC_LON_MAX, lon))
    return lat, lon


# ── Scenario Generators ─────────────────────────────────────────────


def _base_year() -> int:
    return 2016


def generate_request(
    scenario: str = "normal",
    _rng: random.Random = random,
) -> dict:
    """Generate a single prediction request for the given scenario."""
    year = _base_year()

    if scenario == "normal":
        return _gen_normal(_rng, year)
    elif scenario == "rush":
        return _gen_rush_hour_drift(_rng, year)
    elif scenario == "suburban":
        return _gen_suburban_shift(_rng, year)
    elif scenario == "holiday":
        return _gen_holiday_shift(_rng, year)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def _random_dt(
    _rng: random.Random, year: int,
    hour_range: tuple = (0, 23),
    weekday_bias: float = 0.0,
) -> str:
    """Generate a random datetime with optional hour/weekday bias."""
    jan1 = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
    days_offset = _rng.randint(0, 364)
    day_of_week = (jan1 + datetime.timedelta(days=days_offset)).weekday()

    # Slightly bias weekends if weekday_bias > 0 (0.0 = neutral, 1.0 = weekend only)
    if weekday_bias > 0 and _rng.random() < weekday_bias:
        if day_of_week < 5:  # not weekend, nudge forward
            days_offset += 7 - day_of_week

    date = jan1 + datetime.timedelta(days=days_offset)
    hour = _rng.randint(*hour_range)
    minute = _rng.randint(0, 59)
    dt = date.replace(hour=hour, minute=minute)
    return dt.isoformat()


def _random_location(
    _rng: random.Random,
    borough: str,
    spread: float = 0.01,
) -> tuple[float, float]:
    region = BOROUGH_REGIONS.get(borough, BOROUGH_REGIONS["Manhattan"])
    lat = _rng.uniform(*region["lat"])
    lon = _rng.uniform(*region["lon"])
    lat += _rng.gauss(0, spread)
    lon += _rng.gauss(0, spread)
    lat, lon = _clamp_coords(lat, lon)
    return lat, lon


# ─ Scenarios ─────────────────────────────────────────────────────

def _dropoff_near(pickup_lat: float, pickup_lon: float,
                   dist: float, _rng: random.Random) -> tuple[float, float]:
    """Generate dropoff coordinates at a reasonable distance from pickup."""
    drop_lat = pickup_lat + _rng.gauss(0, dist)
    drop_lon = pickup_lon + _rng.gauss(0, dist)
    drop_lat, drop_lon = _clamp_coords(drop_lat, drop_lon)
    return round(drop_lat, 6), round(drop_lon, 6)


def _response_error_detail(resp: requests.Response) -> str:
    """Extract a concise failure detail from FastAPI response."""
    try:
        payload = resp.json()
    except ValueError:
        text = (resp.text or "").strip()
        return text if text else "no response body"

    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            loc = first.get("loc")
            msg = first.get("msg")
            typ = first.get("type")
            return f"loc={loc} msg={msg} type={typ}"
    return str(detail)


def _build_normal_requests(count: int, seed: int) -> list[dict]:
    """Sample valid historical rows and format them as API requests."""
    raw = load_raw(sample_mode=False)
    raw = raw.dropna(subset=RAW_REQUIRED_COLS).copy()

    # Align baseline sampled rows with serving contract to avoid schema failures.
    raw = raw[
        (raw["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX))
        & (raw["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX))
        & (raw["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX))
        & (raw["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX))
        & (raw["passenger_count"].between(PASSENGER_MIN, PASSENGER_MAX))
        & (raw["vendor_id"].between(VENDOR_MIN, VENDOR_MAX))
        & (raw["store_and_fwd_flag"].astype(str).isin(["N", "Y"]))
    ]

    if raw.empty:
        raise ValueError("No valid historical rows available for normal scenario")

    sampled = raw.sample(n=min(count, len(raw)), random_state=seed)
    sampled = sampled[RAW_REQUIRED_COLS].copy()
    sampled["pickup_datetime"] = (
        pd.to_datetime(sampled["pickup_datetime"], utc=True)
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    rows = sampled.to_dict(orient="records")
    if len(rows) < count:
        logger.warning(
            "Normal scenario requested %d rows but only %d valid rows available",
            count,
            len(rows),
        )
    return rows


def _gen_normal(_rng: random.Random, year: int) -> dict:
    """Matches training data distribution roughly."""
    borough = _rng.choice(list(BOROUGH_REGIONS.keys()))
    lat, lon = _random_location(_rng, borough)
    dist = _rng.uniform(0.05, 0.15)  # moderate trip radii
    dlat, dlon = _dropoff_near(lat, lon, dist, _rng)

    return {
        "pickup_latitude": round(lat, 6),
        "pickup_longitude": round(lon, 6),
        "dropoff_latitude": dlat,
        "dropoff_longitude": dlon,
        "passenger_count": _rng.choices([1, 2, 3, 4, 5, 6], weights=[40, 30, 15, 10, 4, 1])[0],
        "borough": borough,
        "service_type": _rng.choice(SERVICE_OPTIONS),
        "vendor_id": _rng.choice([1, 2]),
        "store_and_fwd_flag": _rng.choices(["N", "Y"], weights=[95, 5])[0],
        "pickup_datetime": _random_dt(_rng, year),
    }


def _gen_rush_hour_drift(_rng: random.Random, year: int) -> dict:
    """
    Simulates post-pandemic "staggered rush" — trips concentrate
    in extended peak windows (6-11AM, 4-9PM), with longer average distances.
    """
    borough = _rng.choices(
        list(BOROUGH_REGIONS.keys()),
        weights=[30, 40, 30],  # less Manhattan
    )[0]
    lat, lon = _random_location(_rng, borough, spread=0.03)
    dist = _rng.uniform(0.10, 0.25)  # longer trips
    dlat, dlon = _dropoff_near(lat, lon, dist, _rng)

    # Extended rush hours
    hour_pool = list(range(6, 11)) + list(range(16, 21))
    hour = _rng.choice(hour_pool)
    dt_str = _random_dt(_rng, year, hour_range=(0, 0))
    dt = datetime.datetime.fromisoformat(dt_str).replace(hour=hour)

    return {
        "pickup_latitude": round(lat, 6),
        "pickup_longitude": round(lon, 6),
        "dropoff_latitude": dlat,
        "dropoff_longitude": dlon,
        "passenger_count": _rng.choices([1, 2, 3, 4], weights=[30, 35, 25, 10])[0],
        "borough": borough,
        "service_type": _rng.choices(SERVICE_OPTIONS, weights=[30, 40, 30])[0],
        "vendor_id": _rng.choice([1, 2]),
        "store_and_fwd_flag": _rng.choices(["N", "Y"], weights=[95, 5])[0],
        "pickup_datetime": dt.isoformat(),
    }


def _gen_suburban_shift(_rng: random.Random, year: int) -> dict:
    """
    Simulates suburban expansion — more pickups from Queens/Brooklyn,
    larger drop-off radii, fewer riders per trip.
    """
    borough = _rng.choices(
        list(BOROUGH_REGIONS.keys()),
        weights=[15, 45, 40],
    )[0]
    lat, lon = _random_location(_rng, borough, spread=0.04)
    dist = _rng.uniform(0.12, 0.30)  # significantly longer trips
    dlat, dlon = _dropoff_near(lat, lon, dist, _rng)

    return {
        "pickup_latitude": round(lat, 6),
        "pickup_longitude": round(lon, 6),
        "dropoff_latitude": dlat,
        "dropoff_longitude": dlon,
        "passenger_count": _rng.choices([1, 2, 3], weights=[50, 35, 15])[0],
        "vendor_id": _rng.choice([1, 2]),
        "store_and_fwd_flag": _rng.choices(["N", "Y"], weights=[95, 5])[0],
        "borough": borough,
        "service_type": _rng.choices(SERVICE_OPTIONS, weights=[20, 40, 40])[0],
        "pickup_datetime": _random_dt(_rng, year),
    }


def _gen_holiday_shift(_rng: random.Random, year: int) -> dict:
    """
    Simulates holiday/weekend traffic — short local trips,
    concentrated on weekends, single passengers or pairs.
    """
    borough = _rng.choices(
        list(BOROUGH_REGIONS.keys()),
        weights=[60, 25, 15],  # Manhattan-heavy for tourist traffic
    )[0]
    lat, lon = _random_location(_rng, borough, spread=0.01)
    dist = _rng.uniform(0.02, 0.08)  # short trips
    dlat, dlon = _dropoff_near(lat, lon, dist, _rng)

    return {
        "pickup_latitude": round(lat, 6),
        "pickup_longitude": round(lon, 6),
        "dropoff_latitude": dlat,
        "dropoff_longitude": dlon,
        "passenger_count": _rng.choices([1, 2], weights=[55, 45])[0],
        "vendor_id": _rng.choice([1, 2]),
        "store_and_fwd_flag": _rng.choices(["N", "Y"], weights=[95, 5])[0],
        "borough": borough,
        "service_type": _rng.choices(SERVICE_OPTIONS, weights=[70, 25, 5])[0],
        "pickup_datetime": _random_dt(_rng, year, weekday_bias=0.8),
    }


# ── HTTP Sending ─────────────────────────────────────────────────


def send_batch(
    size: int = 25,
    scenario: str = "normal",
    rng: random.Random | None = None,
    requests_payload: list[dict] | None = None,
) -> list[float]:
    """Send a batch of requests and return latency stats."""
    latencies = []
    if rng is None:
        rng = random.Random(DEFAULT_RANDOM_SEED)

    batch_requests = requests_payload
    if batch_requests is None:
        batch_requests = [generate_request(scenario=scenario, _rng=rng) for _ in range(size)]

    for i, req in enumerate(batch_requests):
        start = time.perf_counter()
        resp = requests.post(API_URL, json=req)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        if resp.status_code == 200:
            eta = resp.json().get("eta_seconds", "N/A")
            logger.debug("Request %2d: LAT=%.1fms | ETA=%.0fs", i + 1, latency_ms, eta)
        else:
            logger.warning(
                "Request %2d: FAILED (%s) - %s",
                i + 1,
                resp.status_code,
                _response_error_detail(resp),
            )

    return latencies


# ── CLI ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate production traffic for drift testing")
    parser.add_argument("--count", type=int, default=75, help="Total requests to send")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size per request wave")
    parser.add_argument(
        "--scenario",
        default="normal",
        choices=["normal", "rush", "suburban", "holiday"],
        help="Traffic distribution scenario (default: normal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for deterministic request generation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seed_everything(args.seed)

    scenario_desc = {
        "normal": "Baseline (historical rows sampled from raw dataset)",
        "rush": "Extended rush hour + longer distances (6-11AM, 4-9PM concentration)",
        "suburban": "Suburban expansion (outer boroughs, longer trip distance, fewer riders)",
        "holiday": "Weekend-heavy, short local Manhattan trips, single passengers",
    }

    print(f"\n🚀 Traffic Simulator ({args.count} requests, scenario='{args.scenario}')")
    print(f"   {scenario_desc[args.scenario]}")
    print("=" * 60)

    all_latencies = []
    rng = random.Random(args.seed)
    normal_rows = _build_normal_requests(args.count, args.seed) if args.scenario == "normal" else None
    for wave in range(0, args.count, args.batch_size):
        batch = min(args.batch_size, args.count - wave)
        print(f"\n🌊 Wave {wave // args.batch_size + 1}...")
        batch_requests = None
        if normal_rows is not None:
            batch_requests = normal_rows[wave: wave + batch]
        lats = send_batch(
            size=batch,
            scenario=args.scenario,
            rng=rng,
            requests_payload=batch_requests,
        )
        all_latencies.extend(lats)

    print("\n" + "=" * 60)
    print(f"✅ Completed {len(all_latencies)} requests")
    print(f"⏱  Avg Latency: {np.mean(all_latencies):.1f}ms")
    print(f"⏱  P95 Latency: {np.percentile(all_latencies, 95):.1f}ms")
    print(f"⏱  Max Latency: {np.max(all_latencies):.1f}ms")
    print(f"\n📊 Check data/monitoring/production_logs.jsonl for logged predictions")
