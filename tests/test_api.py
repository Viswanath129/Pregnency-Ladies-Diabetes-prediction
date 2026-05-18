import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint_success():
    payload = {
        "preg": 6,
        "gluc": 148,
        "bp": 72,
        "skin": 35,
        "ins": 0,
        "bmi": 33.6,
        "dpf": 0.627,
        "age": 50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_percent" in data
    assert "risk_label" in data
    assert "uncertainty" in data
    assert "streams" in data
    assert "classical" in data["streams"]
    assert "ann" in data["streams"]
    assert "stochastic_variance" in data["streams"]

def test_predict_endpoint_invalid_data():
    payload = {
        "preg": "invalid",
        "gluc": 148
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)

def test_predict_endpoint_missing_fields():
    payload = {
        "preg": 6,
        "gluc": 148
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
