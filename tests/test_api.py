import pytest
from fastapi.testclient import TestClient

from app.main import app

# Module-scoped fixture: entering TestClient as a context manager triggers
# FastAPI's startup event (which loads the model/encoder/explainer
# artifacts) — a common gotcha when testing FastAPI apps, since a bare
# TestClient(app) does NOT fire startup/shutdown events on its own.
_client_cm = TestClient(app)
client = _client_cm.__enter__()

VALID_PAYLOAD = {
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2500,
    "DAYS_ID_PUBLISH": -3000,
    "DAYS_LAST_PHONE_CHANGE": -500,
    "CNT_CHILDREN": 1,
    "EXT_SOURCE_1": 0.55,
    "EXT_SOURCE_2": 0.60,
    "EXT_SOURCE_3": 0.50,
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_EDUCATION_TYPE": "Higher education",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "auc" in body
    assert 0 <= body["auc"] <= 1


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"APPROVE", "REVIEW", "REJECT"}
    assert 300 <= body["risk_score"] <= 850
    assert len(body["top_reasons"]) > 0


def test_predict_rejects_missing_field():
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["AMT_INCOME_TOTAL"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_category():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["CODE_GENDER"] = "X"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_income():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["AMT_INCOME_TOTAL"] = -1000
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
