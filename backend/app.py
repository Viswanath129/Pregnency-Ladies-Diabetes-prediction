import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "ensemble": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}

MODELS: Dict[str, Any] = {}
MODELS_LOADED = False

def load_models():
    global MODELS_LOADED
    try:
        for key, filename in FILES.items():
            path = os.path.join(BASE_DIR, filename)
            if not os.path.exists(path):
                logger.warning(f"Model file not found: {path}")
                continue
            MODELS[key] = joblib.load(path)

        if len(MODELS) == len(FILES):
            MODELS_LOADED = True
            logger.info("All models loaded successfully.")
        else:
            logger.warning("Some models failed to load. Simulation mode enabled.")
    except Exception as e:
        logger.error(f"Error loading models: {e}. Simulation mode enabled.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
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
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(index_path)

@app.post("/predict")
async def predict_risk(data: PatientVitals):
    try:
        # 1. Prepare Data
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        if MODELS_LOADED:
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = MODELS["scaler"].transform(df)

            # Get probabilities from individual streams
            p_ml = MODELS["ml"].predict_proba(scaled_data)[:, 1][0]
            
            # ANN prediction
            try:
                p_ann = MODELS["ann"].predict_proba(scaled_data)[:, 1][0]
            except (AttributeError, IndexError):
                pred = MODELS["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Stochastic uncertainty variance
            p_stochastic = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Ensemble aggregation
            ensemble_input = np.array([[p_ml, p_ann, p_stochastic]])
            final_prob = MODELS["ensemble"].predict_proba(ensemble_input)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_stochastic = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_stochastic, is_sim)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

def build_response(final_prob, p_ml, p_ann, p_stochastic, is_sim):
    risk_pct = round(float(final_prob) * 100, 2)
    # Thresholds: Low < 40%, Moderate 40-70%, High > 70%
    label = "High" if risk_pct > 70 else ("Moderate" if risk_pct > 40 else "Low")
    
    return {
        "risk_percent": risk_pct,
        "risk_label": label,
        "uncertainty": round(float(np.std([p_ml, p_ann, p_stochastic])), 4),
        "streams": {
            "classical": round(p_ml * 100, 2),
            "ann": round(p_ann * 100, 2),
            "stochastic_variance": round(p_stochastic * 100, 2)
        },
        "is_simulated": is_sim
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
