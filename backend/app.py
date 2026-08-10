import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
import joblib

# Set up logging to use professional standard logging for diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend_api")

# --- CORS Settings ---
CORS_ORIGINS = ["*"]

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models on startup and save them in app.state.models
    app.state.models = {}
    app.state.models_loaded = False

    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = {
        "ml": "classical_stream.joblib",
        "ann": "ann_stream.joblib",
        "meta": "meta_ai_decision.joblib",
        "scaler": "data_scaler.joblib"
    }

    try:
        app.state.models["ml"] = joblib.load(os.path.join(base_dir, files["ml"]))
        app.state.models["ann"] = joblib.load(os.path.join(base_dir, files["ann"]))
        app.state.models["meta"] = joblib.load(os.path.join(base_dir, files["meta"]))
        app.state.models["scaler"] = joblib.load(os.path.join(base_dir, files["scaler"]))
        app.state.models_loaded = True
        logger.info("All model/scaler joblib assets loaded successfully.")
    except Exception as e:
        logger.error("Could not load models or scaler: %s. Fallback simulation mode enabled.", e)

    yield

    # Cleanup on shutdown
    app.state.models.clear()
    logger.info("Lifespan cleanup complete.")

# Instantiate FastAPI application with the lifespan manager
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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

# Serve the main clinical user interface
@app.get("/")
async def serve_ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "index.html"))

@app.post("/predict")
async def predict_risk(data: PatientVitals):
    try:
        # Prepare Data in the correct order for the scaler
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        models_loaded = getattr(app.state, "models_loaded", False)
        models = getattr(app.state, "models", {})

        if models_loaded and models:
            # Scale input using pandas DataFrame to support feature names
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = models["scaler"].transform(df)

            # Ensure prediction models receive raw numpy arrays to suppress feature name warnings
            if isinstance(scaled_data, pd.DataFrame):
                scaled_data_np = scaled_data.to_numpy()
            else:
                scaled_data_np = np.asarray(scaled_data)

            # Get probabilities from classical stream
            p_ml = models["ml"].predict_proba(scaled_data_np)[:, 1][0]
            
            # ANN prediction (Handling potential different formats)
            try:
                p_ann = models["ann"].predict_proba(scaled_data_np)[:, 1][0]
            except Exception:
                pred = models["ann"].predict(scaled_data_np)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Stochastic Variance (formerly Simulated Quantum variance)
            p_sv = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Ensemble Decision Engine (formerly Meta-AI decision)
            # Pass input as a raw numpy array to avoid feature name warnings
            ensemble_input_np = np.array([[p_ml, p_ann, p_sv]], dtype=np.float32)
            final_prob = models["meta"].predict_proba(ensemble_input_np)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_sv = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_sv, is_sim)

    except Exception as e:
        logger.exception("Inference error occurred in /predict endpoint.")
        # Sanitize internal execution exceptions to avoid leaking internal tracebacks
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the clinical prediction."
        )

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
    # Load configuration from environment variables, fallback to defaults
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))

    uvicorn.run("backend.app:app", host=host, port=port, reload=False)
