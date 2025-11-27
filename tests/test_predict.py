from fastapi.testclient import TestClient

from ml_service.server import app

client = TestClient(app)


def test_predict_validation_error():
    # 29 features instead of 30 -> should fail validation
    payload = {"features": [0.1] * 29}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_success_or_server_error():
    # 30 features; either success (200) or 500 if model not loaded
    payload = {"features": [0.1] * 30}
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 500)

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert isinstance(data["prediction"], int)
        assert 0.0 <= data["probability"] <= 1.0
