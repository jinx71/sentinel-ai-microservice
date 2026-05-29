"""Tests for /predict and /explain, including input validation."""

from fastapi.testclient import TestClient

from tests.conftest import ANOMALOUS_READING, NORMAL_READING


def test_predict_normal_reading(client: TestClient) -> None:
    resp = client.post("/api/v1/predict", json={"readings": [NORMAL_READING]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["count"] == 1
    result = data["results"][0]
    assert result["is_anomaly"] is False
    assert result["severity"] == "normal"


def test_predict_flags_obvious_anomaly(client: TestClient) -> None:
    resp = client.post("/api/v1/predict", json={"readings": [ANOMALOUS_READING]})
    assert resp.status_code == 200
    result = resp.json()["data"]["results"][0]
    assert result["is_anomaly"] is True
    assert result["severity"] in {"low", "medium", "high"}
    assert result["anomaly_score"] > 0


def test_predict_batch_mixed(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/predict",
        json={"readings": [NORMAL_READING, ANOMALOUS_READING]},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["count"] == 2
    assert data["anomalies"] >= 1


def test_predict_rejects_empty_batch(client: TestClient) -> None:
    resp = client.post("/api/v1/predict", json={"readings": []})
    assert resp.status_code == 422


def test_predict_rejects_out_of_range(client: TestClient) -> None:
    bad = {**NORMAL_READING, "temperature": 9999.0}
    resp = client.post("/api/v1/predict", json={"readings": [bad]})
    assert resp.status_code == 422


def test_predict_rejects_unknown_field(client: TestClient) -> None:
    bad = {**NORMAL_READING, "surprise": 1}
    resp = client.post("/api/v1/predict", json={"readings": [bad]})
    assert resp.status_code == 422


def test_explain_uses_rule_based_when_llm_disabled(client: TestClient) -> None:
    resp = client.post("/api/v1/explain", json={"reading": ANOMALOUS_READING})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["source"] == "rule_based"
    assert data["is_anomaly"] is True
    assert len(data["explanation"]) > 0


def test_explain_normal_reading(client: TestClient) -> None:
    resp = client.post("/api/v1/explain", json={"reading": NORMAL_READING})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["is_anomaly"] is False
    assert "nominal" in data["explanation"].lower()
