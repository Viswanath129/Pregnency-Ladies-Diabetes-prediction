import time
import pandas as pd
import numpy as np
import warnings
from backend.app import MODELS_LOADED, MODELS, PatientVitals
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

if not MODELS_LOADED:
    print("Models not loaded. Cannot test actual performance.")
    exit()

def test_inference(use_df=True):
    vitals = [1, 120, 70, 20, 80, 30, 0.5, 25]
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]

    start = time.perf_counter()
    for _ in range(1000):
        if use_df:
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = MODELS["scaler"].transform(df)

            p_ml = MODELS["ml"].predict_proba(scaled_data)[:, 1][0]
            try:
                p_ann = MODELS["ann"].predict_proba(scaled_data)[:, 1][0]
            except:
                pred = MODELS["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]
            p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
            final_prob = MODELS["meta"].predict_proba(meta_input)[:, 1][0]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                vitals_arr = np.array([vitals])
                scaled_data = MODELS["scaler"].transform(vitals_arr)

                p_ml = float(MODELS["ml"].predict_proba(scaled_data)[:, 1][0])

                try:
                    p_ann = float(MODELS["ann"].predict_proba(scaled_data)[:, 1][0])
                except:
                    pred = MODELS["ann"].predict(scaled_data)
                    p_ann = float(pred[0][0] if len(pred.shape) > 1 else pred[0])

                p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))

                meta_input_arr = np.array([[p_ml, p_ann, p_q]])
                final_prob = float(MODELS["meta"].predict_proba(meta_input_arr)[:, 1][0])
    end = time.perf_counter()
    return end - start

df_time = test_inference(True)
print(f"Time with pandas DataFrame: {df_time:.4f}s")
arr_time = test_inference(False)
print(f"Time with numpy array: {arr_time:.4f}s")
