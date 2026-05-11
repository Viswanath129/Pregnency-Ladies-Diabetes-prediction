from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
data = {"preg": 2, "gluc": 120, "bp": 70, "skin": 20, "ins": 80, "bmi": 25.5, "dpf": 0.5, "age": 30}
response = client.post("/predict", json=data)
print(response.json())
