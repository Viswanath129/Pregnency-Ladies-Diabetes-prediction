from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import uvicorn
import joblib
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MODEL LOADING WITH LIFESPAN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "ensemble": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.models_loaded = False
    try:
        app.state.models["ml"] = joblib.load(os.path.join(BASE_DIR, FILES["ml"]))
        app.state.models["ann"] = joblib.load(os.path.join(BASE_DIR, FILES["ann"]))
        app.state.models["ensemble"] = joblib.load(os.path.join(BASE_DIR, FILES["ensemble"]))
        app.state.models["scaler"] = joblib.load(os.path.join(BASE_DIR, FILES["scaler"]))
        app.state.models_loaded = True
        logger.info("All .joblib models loaded successfully.")
    except Exception as e:
        logger.warning(f"Warning: Could not load models ({e}). Simulation mode enabled.")
    yield
    app.state.models.clear()

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
async def predict_risk(data: PatientVitals):
    try:
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        if app.state.models_loaded:
            # Use DataFrame for components fitted with feature names (the scaler)
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = app.state.models["scaler"].transform(df)

            # Ensure inputs to models fitted without feature names are raw numpy arrays
            # scaled_data is already a numpy array from scaler.transform(df)

            p_ml = app.state.models["ml"].predict_proba(scaled_data)[:, 1][0]
            
            try:
                p_ann = app.state.models["ann"].predict_proba(scaled_data)[:, 1][0]
            except Exception:
                pred = app.state.models["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Stochastic Variance (formerly Quantum)
            p_sv = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Ensemble Decision Engine
            # Models fitted without feature names require raw numpy arrays
            ensemble_input = np.array([[p_ml, p_ann, p_sv]])
            final_prob = app.state.models["ensemble"].predict_proba(ensemble_input)[:, 1][0]
            is_sim = False
        else:
            # Fallback Simulation
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_sv = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_sv, is_sim)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

def build_response(final_prob, p_ml, p_ann, p_sv, is_sim):
    risk_pct = round(float(final_prob) * 100, 2)
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
