from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_predict_endpoint():
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
