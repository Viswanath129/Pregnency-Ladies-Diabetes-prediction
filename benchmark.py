import time
import pandas as pd
import numpy as np
import random
import math
import warnings
import joblib

class MockScaler:
    def transform(self, x):
        return x

class MockModel:
    def predict_proba(self, x):
        return np.array([[0.1, 0.9]])

    def predict(self, x):
        return np.array([[0.9]])

scaler = MockScaler()
ml = MockModel()
ann = MockModel()
meta = MockModel()

vitals = [1.0, 100.0, 70.0, 20.0, 80.0, 25.0, 0.5, 30.0]
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]

def old_way():
    df = pd.DataFrame([vitals], columns=cols)
    scaled_data = scaler.transform(df)
    p_ml = ml.predict_proba(scaled_data)[:, 1][0]
    p_ann = ann.predict_proba(scaled_data)[:, 1][0]
    p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)
    meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
    final_prob = meta.predict_proba(meta_input)[:, 1][0]

    risk_pct = round(float(final_prob) * 100, 2)
    label = "High" if risk_pct > 70 else ("Moderate" if risk_pct > 40 else "Low")
    uncert = round(float(np.std([p_ml, p_ann, p_q])), 4)

    return final_prob

def new_way():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = np.array([vitals])
        scaled_data = scaler.transform(arr)
        p_ml = float(ml.predict_proba(scaled_data)[0, 1])
        p_ann = float(ann.predict_proba(scaled_data)[0, 1])
        p_q = min(max(p_ml + random.gauss(0, 0.02), 0.0), 1.0)

        meta_input = np.array([[p_ml, p_ann, p_q]])
        final_prob = float(meta.predict_proba(meta_input)[0, 1])

    risk_pct = round(final_prob * 100, 2)
    label = "High" if risk_pct > 70 else ("Moderate" if risk_pct > 40 else "Low")

    # manual std
    mean = (p_ml + p_ann + p_q) / 3.0
    var = ((p_ml - mean)**2 + (p_ann - mean)**2 + (p_q - mean)**2) / 3.0
    uncert = round(math.sqrt(var), 4)

    return final_prob

# warmup
for _ in range(100):
    old_way()
    new_way()

t0 = time.time()
for _ in range(1000):
    old_way()
t1 = time.time()
print("Old:", t1 - t0)

t0 = time.time()
for _ in range(1000):
    new_way()
t1 = time.time()
print("New:", t1 - t0)
