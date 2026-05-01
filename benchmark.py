import timeit

setup = """
import numpy as np
import pandas as pd
import warnings

# Mock models
class MockScaler:
    def transform(self, X):
        return X

class MockModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])

class MockANN:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])

class MockMeta:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])

scaler = MockScaler()
ml = MockModel()
ann = MockANN()
meta = MockMeta()
vitals = [1.0, 120.0, 70.0, 20.0, 80.0, 25.0, 0.5, 30.0]
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
"""

pandas_code = """
df = pd.DataFrame([vitals], columns=cols)
scaled_data = scaler.transform(df)
p_ml = float(ml.predict_proba(scaled_data)[:, 1][0])
p_ann = float(ann.predict_proba(scaled_data)[:, 1][0])
p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))
meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
final_prob = float(meta.predict_proba(meta_input)[:, 1][0])
"""

numpy_code = """
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    arr = np.array([vitals])
    scaled_data = scaler.transform(arr)
    p_ml = float(ml.predict_proba(scaled_data)[:, 1][0])
    p_ann = float(ann.predict_proba(scaled_data)[:, 1][0])
    p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))
    meta_input = np.array([[p_ml, p_ann, p_q]])
    final_prob = float(meta.predict_proba(meta_input)[:, 1][0])
"""

pandas_time = timeit.timeit(pandas_code, setup=setup, number=1000)
numpy_time = timeit.timeit(numpy_code, setup=setup, number=1000)

print(f"Pandas approach: {pandas_time:.4f} seconds per 1000 iterations")
print(f"Numpy approach: {numpy_time:.4f} seconds per 1000 iterations")
print(f"Improvement: {(pandas_time - numpy_time) / pandas_time * 100:.2f}% faster")
