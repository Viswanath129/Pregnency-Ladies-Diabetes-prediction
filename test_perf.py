import time
import requests
import multiprocessing
import subprocess
import os
import sys

url = "http://127.0.0.1:8000/predict"
data = {"preg": 2, "gluc": 120, "bp": 70, "skin": 20, "ins": 80, "bmi": 25.5, "dpf": 0.5, "age": 30}

def single_request():
    start = time.time()
    response = requests.post(url, json=data)
    end = time.time()
    return end - start

def worker(n):
    return [single_request() for _ in range(n)]

def run_benchmark():
    n_requests = 100
    n_workers = 10

    # Wait for server to be up
    for _ in range(30):
        try:
            requests.get("http://127.0.0.1:8000/")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    else:
        print("Server did not start in time")
        sys.exit(1)

    requests.post(url, json=data) # warmup
    start_total = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(worker, [n_requests] * n_workers)
    end_total = time.time()
    times = [item for sublist in results for item in sublist]
    print(f"Total time: {end_total - start_total:.2f}s")
    print(f"Requests per second: {len(times) / (end_total - start_total):.2f}")

if __name__ == "__main__":
    # Start server
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"])
    try:
        run_benchmark()
    finally:
        proc.terminate()
