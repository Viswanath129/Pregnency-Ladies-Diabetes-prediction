import pandas as pd
import numpy as np
import time
import warnings
import joblib

MODELS = {}
try:
    MODELS["ml"] = joblib.load("backend/classical_stream.joblib")
    MODELS["ann"] = joblib.load("backend/ann_stream.joblib")
    MODELS["meta"] = joblib.load("backend/meta_ai_decision.joblib")
    MODELS["scaler"] = joblib.load("backend/data_scaler.joblib")
except Exception as e:
    print(e)
    exit()

vitals = [1.0, 120.0, 70.0, 20.0, 80.0, 25.0, 0.5, 30.0]
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]

def predict_pandas():
    df = pd.DataFrame([vitals], columns=cols)
    scaled_data = MODELS["scaler"].transform(df)
    p_ml = MODELS["ml"].predict_proba(scaled_data)[:, 1][0]
    p_ann = MODELS["ann"].predict_proba(scaled_data)[:, 1][0]
    p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)
    meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
    final_prob = MODELS["meta"].predict_proba(meta_input)[:, 1][0]
    return final_prob

def predict_numpy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arr = np.array([vitals])
        scaled_data = MODELS["scaler"].transform(arr)
        p_ml = float(MODELS["ml"].predict_proba(scaled_data)[:, 1][0])
        p_ann = float(MODELS["ann"].predict_proba(scaled_data)[:, 1][0])
        p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))
        meta_input = np.array([[p_ml, p_ann, p_q]])
        final_prob = float(MODELS["meta"].predict_proba(meta_input)[:, 1][0])
    return final_prob

# Warmup
predict_pandas()
predict_numpy()

import timeit
t_pandas = timeit.timeit(predict_pandas, number=100)
t_numpy = timeit.timeit(predict_numpy, number=100)

print(f"Pandas: {t_pandas:.4f}s")
print(f"Numpy: {t_numpy:.4f}s")
