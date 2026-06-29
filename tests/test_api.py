import pytest
from fastapi.testclient import TestClient
from backend.app import app
import numpy as np

def test_predict_endpoint_simulation():
    # Test simulation mode (or real mode if models load)
    with TestClient(app) as client:
        payload = {
            "preg": 2,
            "gluc": 120,
            "bp": 70,
            "skin": 20,
            "ins": 80,
            "bmi": 25.0,
            "dpf": 0.5,
            "age": 30
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_percent" in data
        assert "risk_label" in data
        assert "streams" in data
        assert "classical" in data["streams"]
        assert "ann" in data["streams"]
        assert "stochastic" in data["streams"]

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
