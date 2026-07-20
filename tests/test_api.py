import pytest
from fastapi.testclient import TestClient
from backend.app import app

def test_serve_ui():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Risk2" in response.text

def test_predict_risk_success():
    with TestClient(app) as client:
        payload = {
            "preg": 2.0,
            "gluc": 110.0,
            "bp": 70.0,
            "skin": 20.0,
            "ins": 80.0,
            "bmi": 26.5,
            "dpf": 0.45,
            "age": 35.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_percent" in data
        assert "risk_label" in data
        assert "uncertainty" in data
        assert "streams" in data
        assert "is_simulated" in data
        assert data["is_simulated"] is False

        streams = data["streams"]
        assert "classical" in streams
        assert "ann" in streams
        assert "stochastic_variance" in streams

def test_predict_risk_simulation_fallback():
    with TestClient(app) as client:
        client.app.state.models_loaded = False
        payload = {
            "preg": 2.0,
            "gluc": 110.0,
            "bp": 70.0,
            "skin": 20.0,
            "ins": 80.0,
            "bmi": 26.5,
            "dpf": 0.45,
            "age": 35.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_simulated"] is True
        assert "risk_percent" in data
        assert "risk_label" in data

def test_predict_risk_sanitized_error():
    with TestClient(app) as client:
        if hasattr(client.app.state, "models") and "scaler" in client.app.state.models:
            original_scaler = client.app.state.models["scaler"]

            class ErrorScaler:
                def transform(self, df):
                    raise RuntimeError("Sensitive database/internal exception detail")

            client.app.state.models["scaler"] = ErrorScaler()

            payload = {
                "preg": 2.0,
                "gluc": 110.0,
                "bp": 70.0,
                "skin": 20.0,
                "ins": 80.0,
                "bmi": 26.5,
                "dpf": 0.45,
                "age": 35.0
            }
            try:
                response = client.post("/predict", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert "error" in data
                assert "Sensitive database/internal exception detail" not in data["error"]
                assert "internal error occurred" in data["error"]
            finally:
                client.app.state.models["scaler"] = original_scaler
