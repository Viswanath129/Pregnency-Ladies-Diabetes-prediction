import time
import requests
import multiprocessing
import numpy as np
from pydantic import BaseModel

url = "http://127.0.0.1:8000/predict"

data = {
    "preg": 2,
    "gluc": 120,
    "bp": 70,
    "skin": 20,
    "ins": 80,
    "bmi": 25.5,
    "dpf": 0.5,
    "age": 30
}

def single_request():
    start = time.time()
    response = requests.post(url, json=data)
    end = time.time()
    return end - start

def worker(n):
    return [single_request() for _ in range(n)]

if __name__ == "__main__":
    n_requests = 100
    n_workers = 10

    print("Warming up...")
    requests.post(url, json=data)

    print(f"Running {n_requests * n_workers} requests across {n_workers} workers...")

    start_total = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(worker, [n_requests] * n_workers)
    end_total = time.time()

    times = [item for sublist in results for item in sublist]

    print(f"Total time: {end_total - start_total:.2f}s")
    print(f"Requests per second: {len(times) / (end_total - start_total):.2f}")
    print(f"Average latency: {np.mean(times)*1000:.2f}ms")
    print(f"p95 latency: {np.percentile(times, 95)*1000:.2f}ms")
    print(f"p99 latency: {np.percentile(times, 99)*1000:.2f}ms")
