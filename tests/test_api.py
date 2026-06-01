import pytest
from fastapi.testclient import TestClient
from backend.app import app
import numpy as np

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_read_main(client):
    """Test the root endpoint serving the UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Risk2" in response.text

def test_predict_endpoint_success(client):
    """Test a successful prediction request."""
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

    assert isinstance(data["risk_percent"], float)
    assert data["risk_label"] in ["Low", "Moderate", "High"]
    assert "classical" in data["streams"]
    assert "ann" in data["streams"]
    assert "stochastic variance" in data["streams"]

def test_predict_endpoint_invalid_data(client):
    """Test the prediction endpoint with invalid data types."""
    payload = {
        "preg": "not-a-number",
        "gluc": 120.0,
        "bp": 70.0,
        "skin": 20.0,
        "ins": 80.0,
        "bmi": 25.0,
        "dpf": 0.5,
        "age": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_missing_field(client):
    """Test the prediction endpoint with a missing required field."""
    payload = {
        "preg": 2.0,
        "gluc": 120.0,
        "bp": 70.0
        # Missing other fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
