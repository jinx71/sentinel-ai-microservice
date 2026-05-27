"""Direct unit tests for the AnomalyDetector (no HTTP layer)."""

from app.model import AnomalyDetector, generate_synthetic_normal

CONTAMINATION = 0.02
SEED = 42

NORMAL = {
    "equipment_id": "A",
    "temperature": 22.0,
    "pressure": 4.0,
    "vibration": 2.5,
    "humidity": 45.0,
    "flow_rate": 120.0,
}
ANOMALOUS = {
    "equipment_id": "B",
    "temperature": 70.0,
    "pressure": 14.0,
    "vibration": 30.0,
    "humidity": 95.0,
    "flow_rate": 10.0,
}


def test_synthetic_data_shape() -> None:
    x = generate_synthetic_normal(100, SEED)
    assert x.shape == (100, 5)


def test_train_and_predict_roundtrip() -> None:
    det = AnomalyDetector.train(contamination=CONTAMINATION, seed=SEED)
    results = det.predict([NORMAL, ANOMALOUS])
    assert len(results) == 2
    by_id = {r["equipment_id"]: r for r in results}
    assert by_id["A"]["is_anomaly"] is False
    assert by_id["B"]["is_anomaly"] is True


def test_predict_is_deterministic() -> None:
    det1 = AnomalyDetector.train(contamination=CONTAMINATION, seed=SEED)
    det2 = AnomalyDetector.train(contamination=CONTAMINATION, seed=SEED)
    r1 = det1.predict([ANOMALOUS])[0]
    r2 = det2.predict([ANOMALOUS])[0]
    assert r1["anomaly_score"] == r2["anomaly_score"]


def test_save_and_load(tmp_path) -> None:
    path = tmp_path / "m.joblib"
    det = AnomalyDetector.train(contamination=CONTAMINATION, seed=SEED)
    det.save(path)
    assert path.exists()
    loaded = AnomalyDetector.load(path)
    assert loaded.version == det.version


def test_load_or_train_creates_artifact(tmp_path) -> None:
    path = tmp_path / "nested" / "m.joblib"
    det = AnomalyDetector.load_or_train(path, contamination=CONTAMINATION, seed=SEED)
    assert path.exists()
    assert det.version.startswith("iforest")


def test_empty_batch_returns_empty() -> None:
    det = AnomalyDetector.train(contamination=CONTAMINATION, seed=SEED)
    assert det.predict([]) == []
