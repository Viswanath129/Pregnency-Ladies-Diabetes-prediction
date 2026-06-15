from fastapi import FastAPI, Request
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

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "meta": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}

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
        print(f"⚠️ Warning: Could not load models ({e}). Simulation mode enabled.")
        app.state.models_loaded = False
    yield

app = FastAPI(lifespan=lifespan)

# --- CORS Settings ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def predict_risk(data: PatientVitals, request: Request):
    try:
        # 1. Prepare Data in the correct order for the scaler
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        if request.app.state.models_loaded:
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = request.app.state.scaler.transform(df)

            # Get probabilities from individual streams using raw array to suppress feature name warnings
            # scaled_data is already a numpy array from the scaler
            p_ml = request.app.state.ml.predict_proba(scaled_data)[:, 1][0]
            
            # ANN prediction (Handling potential different formats)
            try:
                p_ann = request.app.state.ann.predict_proba(scaled_data)[:, 1][0]
            except:
                pred = request.app.state.ann.predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Stochastic Variance
            p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Final Ensemble Decision Engine
            meta_input = np.array([[p_ml, p_ann, p_q]])
            final_prob = request.app.state.meta.predict_proba(meta_input)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_q = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_q, is_sim)

    except Exception as e:
        return {"error": str(e)}

def build_response(final_prob, p_ml, p_ann, p_q, is_sim):
    risk_pct = round(float(final_prob) * 100, 2)
    # Thresholds: Low < 40%, Moderate 40-70%, High > 70%
    label = "High" if risk_pct > 70 else ("Moderate" if risk_pct > 40 else "Low")
    
    return {
        "risk_percent": risk_pct,
        "risk_label": label,
        "uncertainty": round(float(np.std([p_ml, p_ann, p_q])), 4),
        "streams": {
            "classical": round(p_ml * 100, 2),
            "ann": round(p_ann * 100, 2),
            "stochastic_variance": round(p_q * 100, 2)
        },
        "is_simulated": is_sim
    }

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)