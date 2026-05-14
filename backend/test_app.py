from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_serve_ui():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint():
    payload = {
        "preg": 1,
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
