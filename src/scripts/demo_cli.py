"""
Demo Playbook for NYC ETA API.
Runs a curated set of valid, invalid, and edge-case scenarios against Docker container.
"""
import json
import httpx
import sys

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = httpx.Timeout(10.0)


def api_call(method, path, payload=None, endpoint_desc=""):
    """Helper to send API requests and format results cleanly."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, path, json=payload)

        if response.status_code < 400:
            if endpoint_desc == "health":
                print("   ✅ System healthy\n")
            elif endpoint_desc == "model":
                print("   ✅ Model metadata retrieved:")
                data = response.json()
                print(f"      Model: {data['model_type']}, MAE: {data['metrics']['mae']:.2f}s, Run: {data['run_id'][:8]}...")
            elif endpoint_desc == "single_predict":
                data = response.json()
                print(f"   ✅ ETA: {data['eta_seconds']:.1f} seconds\n")
            elif endpoint_desc == "batch_predict":
                data = response.json()
                results = data.get("predictions", [])
                print(f"   ✅ Batch of {len(results)} trips processed:")
                for i, item in enumerate(results, 1):
                    print(f"      {i}. ETA {item['eta_seconds']:.1f}s")
                print()
            else:
                print(f"   ✅ {response.status_code}: {response.json()}\n")
        else:
            # Expected rejection for invalid demo cases
            data = response.json()
            error_msg = data.get("detail", [])[0].get("msg", "validation error") if isinstance(data, dict) else "validation error"
            print(f"   🛡️ {endpoint_desc} rejected (422): {error_msg}\n")
    except httpx.ConnectError:
        print(f"❌ Cannot connect to {BASE_URL}. Is the Docker container running?\n")
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")


def run_demo():
    print("=" * 60)
    print("🚗 NYC Ride ETA API — Demo Playbook")
    print("=" * 60)

    # --- Health & Metadata ---
    print("\n📡 [System Checks]")
    api_call("GET", f"{BASE_URL}/health", endpoint_desc="health")
    api_call("GET", f"{BASE_URL}/model-info", endpoint_desc="model")

    # --- Valid Scenarios ---
    print("\n🟢 [Valid Single Prediction: Manhattan → Central Park, Friday 6PM]")
    api_call("POST", f"{BASE_URL}/predict", {
        "pickup_datetime": "2016-05-15T18:00:00",
        "pickup_latitude": 40.7128, "pickup_longitude": -74.0060,
        "dropoff_latitude": 40.7831, "dropoff_longitude": -73.9712,
        "passenger_count": 2,
        "vendor_id": 1, "store_and_fwd_flag": "N"
    }, endpoint_desc="single_predict")

    print("🟢 [Valid Batch Prediction: 5 diverse trips]")
    api_call("POST", f"{BASE_URL}/predict/batch", {
        "requests": [
            {"pickup_datetime": "2016-05-15T08:00:00", "pickup_latitude": 40.7128, "pickup_longitude": -74.0060,
             "dropoff_latitude": 40.7831, "dropoff_longitude": -73.9712,
             "passenger_count": 2, "vendor_id": 1, "store_and_fwd_flag": "N"},
            {"pickup_datetime": "2016-05-15T14:30:00", "pickup_latitude": 40.7484, "pickup_longitude": -73.9857,
             "dropoff_latitude": 40.7411, "dropoff_longitude": -73.9897,
             "passenger_count": 1, "vendor_id": 2, "store_and_fwd_flag": "N"},
            {"pickup_datetime": "2016-05-15T23:00:00", "pickup_latitude": 40.6892, "pickup_longitude": -74.1745,
             "dropoff_latitude": 40.7589, "dropoff_longitude": -73.9851,
             "passenger_count": 4, "vendor_id": 1, "store_and_fwd_flag": "N"},
            {"pickup_datetime": "2016-05-15T16:00:00", "pickup_latitude": 40.7527, "pickup_longitude": -73.9772,
             "dropoff_latitude": 40.7769, "dropoff_longitude": -73.9821,
             "passenger_count": 2, "vendor_id": 1, "store_and_fwd_flag": "N"},
            {"pickup_datetime": "2016-05-15T10:00:00", "pickup_latitude": 40.7209, "pickup_longitude": -73.9499,
             "dropoff_latitude": 40.7480, "dropoff_longitude": -73.9855,
             "passenger_count": 1, "vendor_id": 1, "store_and_fwd_flag": "N"},
        ]
    }, endpoint_desc="batch_predict")

    # --- Invalid / Edge Cases ---
    print("\n🔴 [Invalid: Out-of-Range Coordinates (LA instead of NYC)]")
    api_call("POST", f"{BASE_URL}/predict", {
        "pickup_datetime": "2016-05-15T12:00:00",
        "pickup_latitude": 34.0, "pickup_longitude": -118.0, "dropoff_latitude": 34.1, "dropoff_longitude": -118.1,
        "passenger_count": 1, "vendor_id": 1, "store_and_fwd_flag": "N"
    }, endpoint_desc="la_coords")

    print("🔴 [Invalid: Missing Required Field ('pickup_datetime')]")
    api_call("POST", f"{BASE_URL}/predict", {
        "pickup_latitude": 40.7, "pickup_longitude": -74.0, "dropoff_latitude": 40.8, "dropoff_longitude": -73.9,
        "passenger_count": 1, "vendor_id": 1, "store_and_fwd_flag": "N"
    }, endpoint_desc="missing_hour")

    print("🔴 [Invalid: Too Many Passengers (>9)]")
    api_call("POST", f"{BASE_URL}/predict", {
        "pickup_datetime": "2016-05-15T08:00:00",
        "pickup_latitude": 40.7, "pickup_longitude": -74.0, "dropoff_latitude": 40.8, "dropoff_longitude": -73.9,
        "passenger_count": 10, "vendor_id": 1, "store_and_fwd_flag": "N"
    }, endpoint_desc="too_many_passengers")

    print("🔴 [Invalid: Wrong Vendor ID]")
    api_call("POST", f"{BASE_URL}/predict", {
        "pickup_datetime": "2016-05-15T08:00:00",
        "pickup_latitude": 40.7, "pickup_longitude": -74.0, "dropoff_latitude": 40.8, "dropoff_longitude": -73.9,
        "passenger_count": 1, "vendor_id": 5, "store_and_fwd_flag": "N"
    }, endpoint_desc="bad_vendor_id")

    print("=" * 60)
    print("✅ Demo Playbook Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
