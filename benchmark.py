import time
import requests

# We assume the server is running on port 8000
import json

payload = {
  "preg": 2,
  "gluc": 120,
  "bp": 70,
  "skin": 20,
  "ins": 80,
  "bmi": 25,
  "dpf": 0.5,
  "age": 30
}

url = "http://127.0.0.1:8000/predict"
# warm up
for _ in range(10):
    requests.post(url, json=payload)

start = time.time()
for _ in range(100):
    requests.post(url, json=payload)
end = time.time()
print(f"Total time for 100 requests: {end - start:.4f}s")
