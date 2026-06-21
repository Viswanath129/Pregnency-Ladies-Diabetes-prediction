import time
import requests
import json

data = {
    "preg": 1.0,
    "gluc": 150.0,
    "bp": 70.0,
    "skin": 20.0,
    "ins": 79.0,
    "bmi": 25.0,
    "dpf": 0.5,
    "age": 30.0
}

url = "http://127.0.0.1:8000/predict"

def run_bench():
    # warmup
    for _ in range(10):
        requests.post(url, json=data)

    start = time.time()
    for _ in range(500):
        requests.post(url, json=data)
    end = time.time()

    print(f"500 requests took {end - start:.4f} seconds")
    print(f"{(end - start) / 500 * 1000:.4f} ms per request")

if __name__ == "__main__":
    run_bench()
