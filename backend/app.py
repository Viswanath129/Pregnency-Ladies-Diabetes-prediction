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
import logging
from contextlib import asynccontextmanager

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "meta": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models and data scaler during startup
    models = {}
    models_loaded = False
    try:
        models["ml"] = joblib.load(os.path.join(BASE_DIR, FILES["ml"]))
        models["ann"] = joblib.load(os.path.join(BASE_DIR, FILES["ann"]))
        models["meta"] = joblib.load(os.path.join(BASE_DIR, FILES["meta"]))
        models["scaler"] = joblib.load(os.path.join(BASE_DIR, FILES["scaler"]))
        models_loaded = True
        logger.info("All .joblib models loaded successfully.")
    except Exception as e:
        logger.warning("Could not load models (%s). Simulation mode enabled.", e)

    app.state.models = models
    app.state.models_loaded = models_loaded

    yield

    # Shutdown / cleanup
    app.state.models.clear()
    logger.info("Lifespan shutdown: models cleared.")

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
        
        models_loaded = getattr(request.app.state, "models_loaded", False)
        models = getattr(request.app.state, "models", {})

        if models_loaded:
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = models["scaler"].transform(df)

            # Get probabilities from individual streams
            p_ml = models["ml"].predict_proba(scaled_data)[:, 1][0]
            
            # ANN prediction (Handling potential different formats)
            try:
                p_ann = models["ann"].predict_proba(scaled_data)[:, 1][0]
            except:
                pred = models["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Simulated Stochastic Variance
            p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Final Ensemble Decision Engine decision using a raw numpy array to prevent feature name mismatch warning
            meta_input = np.array([[p_ml, p_ann, p_q]])
            final_prob = models["meta"].predict_proba(meta_input)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_q = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_q, is_sim)

    except Exception as e:
        logger.exception("Prediction request failed due to an unexpected exception.")
        return {"error": "An internal error occurred while processing the prediction request."}

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
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
