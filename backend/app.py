from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import uvicorn
import webbrowser
import joblib
from contextlib import asynccontextmanager

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Attempt to load all 4 files from the local directory
    try:
        app.state.ml = joblib.load(os.path.join(BASE_DIR, FILES["ml"]))
        app.state.ann = joblib.load(os.path.join(BASE_DIR, FILES["ann"]))
        app.state.meta = joblib.load(os.path.join(BASE_DIR, FILES["meta"]))
        app.state.scaler = joblib.load(os.path.join(BASE_DIR, FILES["scaler"]))
        app.state.models_loaded = True
        print("✅ All .joblib models loaded successfully.")
    except Exception as e:
        app.state.models_loaded = False
        print(f"⚠️ Warning: Could not load models ({e}). Simulation mode enabled.")
    yield
    # Cleanup
    if hasattr(app.state, 'ml'): del app.state.ml
    if hasattr(app.state, 'ann'): del app.state.ann
    if hasattr(app.state, 'meta'): del app.state.meta
    if hasattr(app.state, 'scaler'): del app.state.scaler

app = FastAPI(lifespan=lifespan)

# --- CORS Settings ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "meta": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}

class PatientVitals(BaseModel):
    preg: float
    gluc: float
    bp: float
    skin: float
    ins: float
    bmi: float
    dpf: float
    age: float

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/predict")
async def predict_risk(data: PatientVitals):
    try:
        # 1. Prepare Data in the correct order for the scaler
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        if app.state.models_loaded:
            # Map standard vitals to training features if necessary
            # Training features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age
            # We use 'DPF' instead of 'DiabetesPedigreeFunction' to match training expectations if needed,
            # though here the cols already use "DPF".

            df = pd.DataFrame([vitals], columns=cols)
            # Use DataFrame to preserve feature names for the scaler
            scaled_data = app.state.scaler.transform(df)

            # Get probabilities from individual streams
            # ML stream
            p_ml = app.state.ml.predict_proba(scaled_data)[:, 1][0]
            
            # ANN prediction (Handling potential different formats)
            try:
                p_ann = app.state.ann.predict_proba(scaled_data)[:, 1][0]
            except:
                pred = app.state.ann.predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Simulated Stochastic Variance
            p_sv = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Final Ensemble Decision Engine
            meta_input = pd.DataFrame([[p_ml, p_ann, p_sv]], columns=['Classical_Prob', 'ANN_Prob', 'Stochastic_Variance_Prob'])
            final_prob = app.state.meta.predict_proba(meta_input.values)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_sv = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_sv, is_sim)

    except Exception as e:
        return {"error": str(e)}

def build_response(final_prob, p_ml, p_ann, p_sv, is_sim):
    risk_pct = round(float(final_prob) * 100, 2)
    # Thresholds: Low < 40%, Moderate 40-70%, High > 70%
    label = "High" if risk_pct > 70 else ("Moderate" if risk_pct > 40 else "Low")
    
    return {
        "risk_percent": risk_pct,
        "risk_label": label,
        "uncertainty": round(float(np.std([p_ml, p_ann, p_sv])), 4),
        "streams": {
            "classical": round(p_ml * 100, 2),
            "ann": round(p_ann * 100, 2),
            "stochastic_variance": round(p_sv * 100, 2)
        },
        "is_simulated": is_sim
    }

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)