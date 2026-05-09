import time
import numpy as np
import pandas as pd
import joblib
import warnings
import os

BASE_DIR = os.path.join(os.getcwd(), "backend")
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "meta": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}

MODELS = {}
try:
    MODELS["ml"] = joblib.load(os.path.join(BASE_DIR, FILES["ml"]))
    MODELS["ann"] = joblib.load(os.path.join(BASE_DIR, FILES["ann"]))
    MODELS["meta"] = joblib.load(os.path.join(BASE_DIR, FILES["meta"]))
    MODELS["scaler"] = joblib.load(os.path.join(BASE_DIR, FILES["scaler"]))
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

vitals = [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0]
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]

def run_pandas():
    df = pd.DataFrame([vitals], columns=cols)
    scaled_data = MODELS["scaler"].transform(df)
    p_ml = MODELS["ml"].predict_proba(scaled_data)[:, 1][0]
    p_ann = MODELS["ann"].predict_proba(scaled_data)[:, 1][0]
    p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)
    meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
    final_prob = MODELS["meta"].predict_proba(meta_input)[:, 1][0]
    return final_prob

def run_numpy():
    vitals_array = np.array([vitals])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaled_data = MODELS["scaler"].transform(vitals_array)
        p_ml = float(MODELS["ml"].predict_proba(scaled_data)[:, 1][0])
        p_ann = float(MODELS["ann"].predict_proba(scaled_data)[:, 1][0])
        p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))
        meta_input = np.array([[p_ml, p_ann, p_q]])
        final_prob = float(MODELS["meta"].predict_proba(meta_input)[:, 1][0])
    return final_prob

n = 1000

t0 = time.time()
for _ in range(n):
    run_pandas()
t_pd = time.time() - t0

t0 = time.time()
for _ in range(n):
    run_numpy()
t_np = time.time() - t0

print(f"Pandas time for {n} iterations: {t_pd:.4f} s")
print(f"Numpy time for {n} iterations:  {t_np:.4f} s")
print(f"Speedup: {t_pd/t_np:.2f}x")
