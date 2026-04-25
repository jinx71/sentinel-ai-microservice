"""Unsupervised anomaly detector for equipment sensor data.

Productionizes the Project 8 model (IsolationForest) behind a small, serialisable
class. The artifact is baked into the image at build time; if absent at runtime the
detector trains itself on synthetic data so the service is never dead-on-arrival.

Why IsolationForest: it is cheap, needs no labels, has a tiny artifact, and starts
in milliseconds — ideal for a containerised microservice that must autoscale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Ordered feature list — the contract between the schema and the model matrix.
FEATURE_ORDER: tuple[str, ...] = (
    "temperature",
    "pressure",
    "vibration",
    "humidity",
    "flow_rate",
)

# Nominal operating envelope (mean, std) for synthetic training data.
_NOMINAL: dict[str, tuple[float, float]] = {
    "temperature": (22.0, 1.5),
    "pressure": (4.0, 0.4),
    "vibration": (2.5, 0.6),
    "humidity": (45.0, 4.0),
    "flow_rate": (120.0, 8.0),
}


def _severity_from_score(score: float, is_anomaly: bool) -> str:
    """Bucket a continuous anomaly score into an operator-facing label."""
    if not is_anomaly:
        return "normal"
    if score >= 0.25:
        return "high"
    if score >= 0.10:
        return "medium"
    return "low"


def generate_synthetic_normal(n_samples: int, seed: int) -> np.ndarray:
    """Sample nominal-operation readings around the operating envelope."""
    rng = np.random.default_rng(seed)
    columns = [
        rng.normal(mean, std, size=n_samples) for mean, std in
        (_NOMINAL[f] for f in FEATURE_ORDER)
    ]
    return np.column_stack(columns)


class AnomalyDetector:
    """Thin wrapper around a fitted IsolationForest."""

    def __init__(self, model: IsolationForest, version: str) -> None:
        self._model = model
        self.version = version

    @classmethod
    def train(cls, contamination: float, seed: int, n_samples: int = 4000) -> AnomalyDetector:
        x = generate_synthetic_normal(n_samples, seed)
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=seed,
            # n_jobs=1: the served model scores tiny batches, so parallelism adds
            # nothing and avoids loky temp-dir writes under a read-only root fs.
            n_jobs=1,
        )
        model.fit(x)
        return cls(model, version=f"iforest-v1-c{contamination}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "version": self.version}, path)
        logger.info("saved model artifact to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> AnomalyDetector:
        payload = joblib.load(path)
        return cls(payload["model"], version=payload["version"])

    @classmethod
    def load_or_train(cls, path: str | Path, contamination: float, seed: int) -> AnomalyDetector:
        """Prefer the baked artifact; fall back to on-the-fly training."""
        path = Path(path)
        if path.exists():
            try:
                detector = cls.load(path)
                logger.info("loaded model artifact %s (version=%s)", path, detector.version)
                return detector
            except Exception:  # noqa: BLE001 - corrupt artifact should not be fatal
                logger.exception("failed to load artifact, retraining")
        logger.warning("artifact missing at %s, training a new model", path)
        detector = cls.train(contamination=contamination, seed=seed)
        try:
            detector.save(path)
        except OSError:
            logger.warning("could not persist freshly trained model (read-only fs?)")
        return detector

    def _to_matrix(self, readings: list[dict]) -> np.ndarray:
        return np.array([[r[f] for f in FEATURE_ORDER] for r in readings], dtype=float)

    def predict(self, readings: list[dict]) -> list[dict]:
        """Score a batch of readings.

        anomaly_score is normalised so higher always means more anomalous, which is
        more intuitive for dashboards than IsolationForest's raw signed output.
        """
        if not readings:
            return []
        x = self._to_matrix(readings)
        # decision_function: positive = normal, negative = anomalous. Negate so the
        # score rises with abnormality, then clip to keep dashboard scales sane.
        raw = -self._model.decision_function(x)
        flags = self._model.predict(x) == -1
        out: list[dict] = []
        for reading, score, is_anom in zip(readings, raw, flags, strict=True):
            score_f = float(np.clip(score, -1.0, 1.0))
            out.append(
                {
                    "equipment_id": reading["equipment_id"],
                    "anomaly_score": round(score_f, 4),
                    "is_anomaly": bool(is_anom),
                    "severity": _severity_from_score(score_f, bool(is_anom)),
                }
            )
        return out
