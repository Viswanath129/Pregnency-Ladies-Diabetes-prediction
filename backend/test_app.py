from fastapi.testclient import TestClient
from backend.app import app
import json

client = TestClient(app)

def test_predict_endpoint():
    data = {"preg": 2, "gluc": 120, "bp": 70, "skin": 20, "ins": 80, "bmi": 25.5, "dpf": 0.5, "age": 30}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    res_data = response.json()
    assert "risk_percent" in res_data
    assert "risk_label" in res_data
    assert "uncertainty" in res_data
    assert "streams" in res_data
    assert "is_simulated" in res_data
