import pytest
from fastapi.testclient import TestClient
from backend.app import app

def test_root_endpoint():
    """Verify that the home page serves the HTML UI successfully."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert b"Risk2" in response.content

def test_predict_endpoint_success_with_models():
    """Test standard valid payload with loaded models."""
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
    with TestClient(app) as client:
        # Check app state is set up and lifespan is triggered
        assert getattr(app.state, "models_loaded", False) is True

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "risk_percent" in data
        assert "risk_label" in data
        assert "uncertainty" in data
        assert "streams" in data
        assert data["is_simulated"] is False

        # Verify streams response keys match Stochastic Variance
        streams = data["streams"]
        assert "classical" in streams
        assert "ann" in streams
        assert "stochastic_variance" in streams
        assert "quantum" not in streams

def test_predict_endpoint_validation_error():
    """Verify that invalid inputs raise standard Pydantic validation errors."""
    payload = {
        "preg": "not-a-number",
        "gluc": 110.0,
        "bp": 70.0,
        "skin": 20.0,
        "ins": 80.0,
        "bmi": 26.5,
        "dpf": 0.45,
        "age": 35.0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 # Unprocessable Entity
        assert "detail" in response.json()

def test_predict_endpoint_fallback_mode(monkeypatch):
    """Ensure system safely degrades to mathematical simulation fallback if models are missing."""
    # Temporarily mark models as not loaded in a test scenario
    with TestClient(app) as client:
        # Clear/Mock state
        monkeypatch.setattr(app.state, "models_loaded", False)

        payload = {
            "preg": 1.0,
            "gluc": 150.0,
            "bp": 80.0,
            "skin": 25.0,
            "ins": 90.0,
            "bmi": 30.0,
            "dpf": 0.5,
            "age": 28.0
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_simulated"] is True
        assert data["risk_percent"] > 0
