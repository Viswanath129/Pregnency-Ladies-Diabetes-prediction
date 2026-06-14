import time
import requests
import multiprocessing
import uvicorn
from backend.app import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

if __name__ == "__main__":
    p = multiprocessing.Process(target=run_server)
    p.start()
    time.sleep(3) # Wait for server to start

    payload = {
        "preg": 2,
        "gluc": 120,
        "bp": 70,
        "skin": 20,
        "ins": 80,
        "bmi": 25.5,
        "dpf": 0.5,
        "age": 30
    }

    try:
        # Warmup
        for _ in range(5):
            requests.post("http://127.0.0.1:8001/predict", json=payload)

        # Benchmark single requests
        start_time = time.time()
        for _ in range(100):
            requests.post("http://127.0.0.1:8001/predict", json=payload)
        end_time = time.time()

        print(f"Time for 100 sequential requests: {end_time - start_time:.4f} seconds")

    finally:
        p.terminate()
        p.join()
