import pytest
from fastapi.testclient import TestClient
from backend.app import app

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Risk2Relief" in response.text

def test_predict_risk_simulation():
    """
    Test the predict endpoint.
    Note: In the test environment, if models are not present, it should fall back to simulation.
    """
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
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_percent" in data
        assert "risk_label" in data
        assert "streams" in data
        assert "stochastic_variance" in data["streams"]
        # Ensure 'quantum' is NOT in the response (updated terminology)
        assert "quantum" not in data["streams"]

def test_predict_invalid_data():
    payload = {
        "preg": "invalid",
        "gluc": 120
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 # Validation error
