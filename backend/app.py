from contextlib import asynccontextmanager
import logging
import os
import uvicorn
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backend.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ml": "classical_stream.joblib",
    "ann": "ann_stream.joblib",
    "meta": "meta_ai_decision.joblib",
    "scaler": "data_scaler.joblib"
}


# --- LIFESPAN CONTEXT MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
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
        logger.warning(f"Could not load models ({e}). Simulation mode enabled.")

    app.state.models = models
    app.state.models_loaded = models_loaded
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
    ui_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="UI file not found.")
    return FileResponse(ui_path)


@app.post("/predict")
async def predict_risk(data: PatientVitals, request: Request):
    try:
        vitals = [
            data.preg, data.gluc, data.bp, data.skin,
            data.ins, data.bmi, data.dpf, data.age
        ]
        cols = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DPF", "Age"
        ]

        models_loaded = getattr(request.app.state, "models_loaded", False)
        models = getattr(request.app.state, "models", {})

        if models_loaded and models:
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = models["scaler"].transform(df)

            # Get probabilities from individual streams
            p_ml = float(models["ml"].predict_proba(scaled_data)[:, 1][0])

            # ANN prediction
            try:
                p_ann = float(models["ann"].predict_proba(scaled_data)[:, 1][0])
            except Exception:
                pred = models["ann"].predict(scaled_data)
                p_ann = float(pred[0][0] if len(pred.shape) > 1 else pred[0])

            # Simulated Quantum variance
            p_q = float(np.clip(p_ml + np.random.normal(0, 0.02), 0, 1))

            # Final Meta-AI decision (pass numpy array to avoid feature name warnings)
            meta_input = np.array([[p_ml, p_ann, p_q]])
            final_prob = float(models["meta"].predict_proba(meta_input)[:, 1][0])
            is_sim = False
        else:
            # Mathematical Simulation fallback
            final_prob = (data.gluc / 300) * 0.7 + (data.bmi / 50) * 0.3
            p_ml, p_ann, p_q = final_prob * 0.9, final_prob * 1.1, final_prob
            is_sim = True

        return build_response(final_prob, p_ml, p_ann, p_q, is_sim)

    except Exception as e:
        logger.error(f"Prediction processing error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred while processing the request."}
        )


def build_response(final_prob, p_ml, p_ann, p_q, is_sim):
    risk_pct = round(float(final_prob) * 100, 2)
    label = "High" if risk_pct > 70 else ("Moderate" if risk_pct > 40 else "Low")

    return {
        "risk_percent": risk_pct,
        "risk_label": label,
        "uncertainty": round(float(np.std([p_ml, p_ann, p_q])), 4),
        "streams": {
            "classical": round(p_ml * 100, 2),
            "ann": round(p_ann * 100, 2),
            "quantum": round(p_q * 100, 2)
        },
        "is_simulated": is_sim
    }


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("backend.app:app", host=host, port=port, reload=False)
