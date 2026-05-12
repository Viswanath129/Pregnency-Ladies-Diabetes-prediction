from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_predict():
    response = client.post(
        "/predict",
        json={
            "preg": 1.0,
            "gluc": 85.0,
            "bp": 66.0,
            "skin": 29.0,
            "ins": 0.0,
            "bmi": 26.6,
            "dpf": 0.351,
            "age": 31.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "risk_percent" in data
    assert "risk_label" in data
    assert "streams" in data

def test_serve_ui():
    response = client.get("/")
    assert response.status_code == 200
