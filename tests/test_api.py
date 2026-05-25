import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_predict_endpoint():
    """
    Test the /predict endpoint with sample patient vitals.
    This test will run in simulation mode if models are not loaded.
    """
    payload = {
        "preg": 2.0,
        "gluc": 120.0,
        "bp": 70.0,
        "skin": 20.0,
        "ins": 80.0,
        "bmi": 25.0,
        "dpf": 0.5,
        "age": 30.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "risk_percent" in data
    assert "risk_label" in data
    assert "uncertainty" in data
    assert "streams" in data
    assert "is_simulated" in data

    # Check if streams reflect the new terminology
    assert "classical" in data["streams"]
    assert "ann" in data["streams"]
    assert "stochastic_variance" in data["streams"]

    # Basic sanity check on values
    assert 0 <= data["risk_percent"] <= 100
    assert data["risk_label"] in ["Low", "Moderate", "High"]

def test_serve_ui():
    """
    Test that the root endpoint serves the UI.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
