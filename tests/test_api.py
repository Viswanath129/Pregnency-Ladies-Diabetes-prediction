import pytest
from fastapi.testclient import TestClient
from backend.app import app

def test_predict_endpoint():
    with TestClient(app) as client:
        payload = {
            "preg": 2,
            "gluc": 120,
            "bp": 70,
            "skin": 20,
            "ins": 80,
            "bmi": 26.5,
            "dpf": 0.45,
            "age": 35
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_percent" in data
        assert "risk_label" in data
        assert "streams" in data
        assert "stochastic_variance" in data["streams"]
