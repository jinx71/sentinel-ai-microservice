"""Health, readiness, root, and metrics endpoint tests."""

from fastapi.testclient import TestClient


def test_liveness(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_ready(client: TestClient) -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["model_version"].startswith("iforest")


def test_root_envelope(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["service"]


def test_metrics_exposed(client: TestClient) -> None:
    # Hit predict first so model counters are populated.
    client.post("/api/v1/predict", json={"readings": [
        {"equipment_id": "X", "temperature": 22.0, "pressure": 4.0,
         "vibration": 2.5, "humidity": 45.0, "flow_rate": 120.0}
    ]})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "predictions_total" in resp.text
    assert "model_loaded" in resp.text
