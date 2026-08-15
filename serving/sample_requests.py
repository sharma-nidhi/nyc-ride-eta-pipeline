"""
Sample API calls for the ETA service (Week 3 / M4) — doubles as submission evidence.

Start the API first (in another terminal):
    uvicorn serving.api:app --port 8000
Then run:
    python serving/sample_requests.py
"""
import requests

URL = "http://127.0.0.1:8000"

valid = {
    "pickup_datetime": "2016-03-14T17:24:00",
    "pickup_longitude": -73.9857, "pickup_latitude": 40.7484,
    "dropoff_longitude": -73.9850, "dropoff_latitude": 40.7580,
    "passenger_count": 1,
}

invalid = {  # latitude off the NYC map + passenger_count too high
    "pickup_datetime": "2016-03-14T17:24:00",
    "pickup_longitude": -73.9857, "pickup_latitude": 99.0,
    "dropoff_longitude": -73.9850, "dropoff_latitude": 40.7580,
    "passenger_count": 20,
}

print("HEALTH :", requests.get(f"{URL}/health").json())

r = requests.post(f"{URL}/predict", json=valid)
print(f"VALID  : {r.status_code} -> {r.json()}")

r = requests.post(f"{URL}/predict", json=invalid)
print(f"INVALID: {r.status_code} -> {r.json()['detail'][0]['loc']} {r.json()['detail'][0]['msg']}")
