import pytest
from fastapi.testclient import TestClient
import joblib
from backend.app import app

def test_serve_ui():
    """Test that the main route serves the HTML UI successfully."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Risk2Relief" in response.text
        assert "Ensemble Decision Engine" in response.text

def test_predict_risk_valid():
    """Test standard prediction pipeline with valid patient inputs."""
    with TestClient(app) as client:
        # Verify app loaded models successfully under normal conditions
        assert client.app.state.models_loaded is True

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

        # Check standard properties
        assert "risk_percent" in data
        assert "risk_label" in data
        assert "uncertainty" in data
        assert "streams" in data
        assert data["is_simulated"] is False

        # Verify stream keys matches modern nomenclature
        streams = data["streams"]
        assert "classical" in streams
        assert "ann" in streams
        assert "stochastic_variance" in streams
        assert "quantum" not in streams

        # Check types
        assert isinstance(data["risk_percent"], float)
        assert 0.0 <= data["risk_percent"] <= 100.0
        assert data["risk_label"] in ["Low", "Moderate", "High"]
        assert isinstance(data["uncertainty"], float)

def test_predict_risk_invalid_data():
    """Test input validation for prediction endpoint with missing and bad fields."""
    with TestClient(app) as client:
        # Scenario A: Missing required parameter (gluc)
        payload_missing = {
            "preg": 2.0,
            "bp": 70.0,
            "skin": 20.0,
            "ins": 80.0,
            "bmi": 26.5,
            "dpf": 0.45,
            "age": 35.0
        }
        response = client.post("/predict", json=payload_missing)
        assert response.status_code == 422

        # Scenario B: Invalid data type for a parameter
        payload_bad_type = {
            "preg": "not_a_number",
            "gluc": 110.0,
            "bp": 70.0,
            "skin": 20.0,
            "ins": 80.0,
            "bmi": 26.5,
            "dpf": 0.45,
            "age": 35.0
        }
        response = client.post("/predict", json=payload_bad_type)
        assert response.status_code == 422

def test_predict_risk_fallback_simulation(monkeypatch):
    """Test fallback simulation mode if serialized models fail to load."""
    # Monkeypatch joblib.load to trigger simulation mode during lifespan model loading
    def mock_load(*args, **kwargs):
        raise RuntimeError("Simulated .joblib model deserialization failure.")

    monkeypatch.setattr(joblib, "load", mock_load)

    with TestClient(app) as client:
        # Lifespan should have run and set state to not loaded
        assert client.app.state.models_loaded is False
        assert len(client.app.state.models) == 0

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
        assert "uncertainty" in data
        assert "streams" in data
