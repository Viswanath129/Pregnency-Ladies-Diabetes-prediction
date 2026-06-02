import time
import requests

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

url = "http://127.0.0.1:8000/predict"

# warmup
for _ in range(10):
    try:
        requests.post(url, json=payload)
    except:
        pass

start = time.time()
n = 500
for _ in range(n):
    requests.post(url, json=payload)
end = time.time()

print(f"Time for {n} requests: {end - start:.4f} seconds")
print(f"Requests per second: {n / (end - start):.2f}")
