import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles model loading and cleanup.
    """
    FILES = {
        "ml": "classical_stream.joblib",
        "ann": "ann_stream.joblib",
        "meta": "meta_ai_decision.joblib",
        "scaler": "data_scaler.joblib"
    }

    app.state.models = {}
    app.state.models_loaded = False

    try:
        # Load all .joblib models from the local directory
        app.state.models["ml"] = joblib.load(os.path.join(BASE_DIR, FILES["ml"]))
        app.state.models["ann"] = joblib.load(os.path.join(BASE_DIR, FILES["ann"]))
        app.state.models["meta"] = joblib.load(os.path.join(BASE_DIR, FILES["meta"]))
        app.state.models["scaler"] = joblib.load(os.path.join(BASE_DIR, FILES["scaler"]))
        app.state.models_loaded = True
        logger.info("Successfully loaded all ensemble decision engine models.")
    except Exception as e:
        logger.warning(f"Could not load models ({e}). Falling back to simulation mode.")

    yield

    # Cleanup
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
async def predict_risk(request: Request, data: PatientVitals):
    try:
        # Prepare Data in the correct order for the scaler
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        if request.app.state.models_loaded:
            # Use DataFrame with feature names for the scaler to suppress warnings
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = request.app.state.models["scaler"].transform(df)

            # Get probabilities from individual streams
            p_ml = request.app.state.models["ml"].predict_proba(scaled_data)[:, 1][0]
            
            # ANN prediction (Handling potential different formats)
            try:
                p_ann = request.app.state.models["ann"].predict_proba(scaled_data)[:, 1][0]
            except (AttributeError, IndexError):
                pred = request.app.state.models["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Stochastic Variance (formerly Quantum stream)
            p_sv = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Final Ensemble Decision Engine decision
            # Ensure input is a raw numpy array as the model was fitted without feature names
            ensemble_input = np.array([[p_ml, p_ann, p_sv]])
            final_prob = request.app.state.models["meta"].predict_proba(ensemble_input)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_sv = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_sv, is_sim)

    except Exception as e:
        logger.exception("Error occurred during risk prediction")
        return {"error": "Internal server error"}

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
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
