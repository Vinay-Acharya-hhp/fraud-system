from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
GOOD = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10.0}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability_and_decision():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud_alert"], bool)


def test_negative_amount_is_rejected():
    assert client.post("/predict", json={**GOOD, "Amount": -5.0}).status_code == 422


def test_missing_field_is_rejected():
    bad = dict(GOOD)
    del bad["V3"]
    assert client.post("/predict", json=bad).status_code == 422
