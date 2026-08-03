from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import os
import uvicorn
import webbrowser
import joblib
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backend")

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = {
        "ml": "classical_stream.joblib",
        "ann": "ann_stream.joblib",
        "meta": "meta_ai_decision.joblib",
        "scaler": "data_scaler.joblib"
    }

    models = {}
    models_loaded = False

    try:
        models["ml"] = joblib.load(os.path.join(base_dir, files["ml"]))
        models["ann"] = joblib.load(os.path.join(base_dir, files["ann"]))
        models["meta"] = joblib.load(os.path.join(base_dir, files["meta"]))
        models["scaler"] = joblib.load(os.path.join(base_dir, files["scaler"]))
        models_loaded = True
        logger.info("All .joblib models loaded successfully.")
    except Exception as e:
        logger.exception("Could not load models. Simulation mode enabled.")

    app.state.models = {
        "models": models,
        "loaded": models_loaded
    }

    yield

    # Cleanup on shutdown
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "index.html"))

@app.post("/predict")
async def predict_risk(data: PatientVitals, request: Request):
    try:
        # 1. Prepare Data in the correct order for the scaler
        vitals = [data.preg, data.gluc, data.bp, data.skin, data.ins, data.bmi, data.dpf, data.age]
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
        
        models_state = getattr(request.app.state, "models", {"loaded": False, "models": {}})
        models_loaded = models_state.get("loaded", False)
        models = models_state.get("models", {})

        if models_loaded:
            # Scaler is fitted with feature names, so use a pandas DataFrame
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = models["scaler"].transform(df)

            # Pass raw numpy array (scaled_data) to models fitted without feature names to suppress warnings
            p_ml = models["ml"].predict_proba(scaled_data)[:, 1][0]
            
            # ANN prediction (Handling potential different formats)
            try:
                p_ann = models["ann"].predict_proba(scaled_data)[:, 1][0]
            except Exception:
                pred = models["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Simulated Stochastic Variance (using professional term instead of 'Quantum')
            p_sv = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Final Ensemble Decision Engine (using professional term instead of 'Meta-AI')
            # Pass a raw numpy array instead of DataFrame to suppress feature name warnings
            ensemble_input = np.array([[p_ml, p_ann, p_sv]], dtype=np.float64)
            final_prob = models["meta"].predict_proba(ensemble_input)[:, 1][0]
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_sv = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_sv, is_sim)

    except Exception as e:
        logger.exception("Error during prediction process")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction processing."
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
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
