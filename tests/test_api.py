import os
import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_serve_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Risk2Relief" in response.text


def test_predict_risk_success(client):
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
    assert data["risk_label"] in ["Low", "Moderate", "High"]
    assert "uncertainty" in data
    assert "streams" in data
    assert "classical" in data["streams"]
    assert "ann" in data["streams"]
    assert "quantum" in data["streams"]
    assert data["is_simulated"] is False


def test_predict_risk_invalid_payload(client):
    payload = {
        "preg": "invalid",
        "gluc": 110.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_risk_simulation_fallback(monkeypatch):
    # Test fallback behavior when models are not loaded
    from backend import app as backend_module

    with TestClient(backend_module.app) as test_client:
        # Simulate missing models
        test_client.app.state.models_loaded = False
        test_client.app.state.models = {}

        payload = {
            "preg": 1.0,
            "gluc": 120.0,
            "bp": 70.0,
            "skin": 20.0,
            "ins": 80.0,
            "bmi": 25.0,
            "dpf": 0.5,
            "age": 30.0
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_simulated"] is True
