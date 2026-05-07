import timeit

setup_code = """
import pandas as pd
import numpy as np
import joblib
import warnings

# Load models
scaler = joblib.load('backend/data_scaler.joblib')
ml = joblib.load('backend/classical_stream.joblib')
ann = joblib.load('backend/ann_stream.joblib')
meta = joblib.load('backend/meta_ai_decision.joblib')

vitals = [1.0, 100.0, 70.0, 20.0, 50.0, 25.0, 0.5, 30.0]
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
"""

pandas_code = """
df = pd.DataFrame([vitals], columns=cols)
scaled_data = scaler.transform(df)

# Get probabilities from individual streams
p_ml = ml.predict_proba(scaled_data)[:, 1][0]

# ANN prediction
try:
    p_ann = ann.predict_proba(scaled_data)[:, 1][0]
except:
    pred = ann.predict(scaled_data)
    p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

# Simulated Quantum variance
p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

# Final Meta-AI decision
meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
final_prob = meta.predict_proba(meta_input)[:, 1][0]
"""

numpy_code = """
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scaled_data = scaler.transform(np.array([vitals]))

    # Get probabilities from individual streams
    p_ml = float(ml.predict_proba(scaled_data)[:, 1][0])

    # ANN prediction
    try:
        p_ann = float(ann.predict_proba(scaled_data)[:, 1][0])
    except:
        pred = ann.predict(scaled_data)
        p_ann = float(pred[0][0] if len(pred.shape) > 1 else pred[0])

    # Simulated Quantum variance
    p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))

    # Final Meta-AI decision
    meta_input = np.array([[p_ml, p_ann, p_q]])
    final_prob = float(meta.predict_proba(meta_input)[:, 1][0])
"""

import timeit
print("Pandas time:", timeit.timeit(stmt=pandas_code, setup=setup_code, number=1000))
print("Numpy time:", timeit.timeit(stmt=numpy_code, setup=setup_code, number=1000))
