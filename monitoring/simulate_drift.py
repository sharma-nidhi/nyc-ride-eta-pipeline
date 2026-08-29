
import random
import requests

URL = "http://127.0.0.1:8000/predict"
random.seed(42)


def jitter(base, spread):
    return round(base + random.uniform(-spread, spread), 4)


def send(n, lat_c, lon_c, spread, hours, passengers):
    ok = 0
    for _ in range(n):
        body = {
            "pickup_datetime": f"2016-06-15T{random.choice(hours):02d}:30:00",
            "pickup_latitude": jitter(lat_c, spread),
            "pickup_longitude": jitter(lon_c, spread),
            "dropoff_latitude": jitter(lat_c, spread),
            "dropoff_longitude": jitter(lon_c, spread),
            "passenger_count": random.choice(passengers),
        }
        if requests.post(URL, json=body).status_code == 200:
            ok += 1
    return ok


# Batch A — NORMAL: short midtown trips, daytime, 1-2 passengers
normal = send(40, 40.758, -73.985, 0.010, hours=[9, 10, 11, 13, 14, 15], passengers=[1, 2])

# Batch B — SURGE: long trips across the city, evening rush, fuller cabs
surge = send(40, 40.730, -73.950, 0.120, hours=[18, 19, 20], passengers=[3, 4, 5])

print(f"Normal batch logged: {normal}")
print(f"Surge batch logged : {surge}")
print("Now run:  python monitoring/check_drift.py")