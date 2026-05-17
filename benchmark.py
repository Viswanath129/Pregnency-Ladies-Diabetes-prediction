import time
import pandas as pd
import warnings

vitals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]

start = time.perf_counter()
for _ in range(1000):
    df = pd.DataFrame([vitals], columns=cols)
end = time.perf_counter()
print(f"DataFrame creation: {end - start:.4f}s")

start = time.perf_counter()
for _ in range(1000):
    arr = [vitals]
end = time.perf_counter()
print(f"List/Array creation: {end - start:.4f}s")
