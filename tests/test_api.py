import pytest
from fastapi.testclient import TestClient
from backend.app import app

def test_predict_endpoint():
    # Use context manager to trigger lifespan events
    with TestClient(app) as client:
        payload = {
            "preg": 2,
            "gluc": 120,
            "bp": 70,
            "skin": 20,
            "ins": 80,
            "bmi": 25.5,
            "dpf": 0.5,
            "age": 30
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_percent" in data
        assert "risk_label" in data
        assert "streams" in data
        assert "stochastic_variance" in data["streams"]
        assert "is_simulated" in data
        # If models loaded correctly, is_simulated should be False
        # (Assuming models are present in the environment)
        assert data["is_simulated"] is False

def test_serve_ui():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

def test_invalid_payload():
    with TestClient(app) as client:
        # Missing required field 'preg'
        payload = {
            "gluc": 120,
            "bp": 70,
            "skin": 20,
            "ins": 80,
            "bmi": 25.5,
            "dpf": 0.5,
            "age": 30
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
