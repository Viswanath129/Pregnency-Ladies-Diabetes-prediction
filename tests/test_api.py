import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.app import app

def test_serve_ui():
    """Verify that the index route serves the UI html page correctly."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Risk2" in response.content

def test_predict_risk_success():
    """Verify that the predict endpoint returns expected fields with correct models."""
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

        # Verify streams terminology
        streams = data["streams"]
        assert "classical" in streams
        assert "ann" in streams
        assert "stochastic_variance" in streams

        # Verify types
        assert isinstance(data["risk_percent"], float)
        assert data["risk_label"] in ["Low", "Moderate", "High"]
        assert isinstance(data["uncertainty"], float)
        assert isinstance(streams["classical"], float)
        assert isinstance(streams["ann"], float)
        assert isinstance(streams["stochastic_variance"], float)
        assert data["is_simulated"] is False

def test_predict_risk_fallback():
    """Verify that the predict endpoint falls back to simulation mode when models are not loaded."""
    with TestClient(app) as client:
        # Manually disable models_loaded in app state
        client.app.state.models["loaded"] = False

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

def test_predict_invalid_data():
    """Verify that validation errors are caught by FastAPI with a 422 status code."""
    with TestClient(app) as client:
        # Missing 'gluc' (glucose) which is a required float
        payload = {
            "preg": 2.0,
            "bp": 70.0,
            "skin": 20.0,
            "ins": 80.0,
            "bmi": 26.5,
            "dpf": 0.45,
            "age": 35.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

def test_predict_exception_handling():
    """Verify that any internal exceptions are properly caught, logged, and sanitized."""
    with TestClient(app) as client:
        # Set up a mock scaler that throws an exception when transform is called
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = Exception("Incompatible scaler dimensions.")

        original_models = client.app.state.models["models"]
        client.app.state.models["models"] = {
            "scaler": mock_scaler,
            "ml": original_models["ml"],
            "ann": original_models["ann"],
            "meta": original_models["meta"]
        }
        client.app.state.models["loaded"] = True

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
        assert response.status_code == 500

        data = response.json()
        # Verify exception message is sanitized (does NOT leak "Incompatible scaler dimensions.")
        assert "detail" in data
        assert "An error occurred during prediction processing." in data["detail"]
        assert "Incompatible scaler dimensions" not in data["detail"]
