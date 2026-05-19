"""Train and persist the anomaly model.

Run during the Docker build so the image ships with a ready artifact:
    python scripts/train_model.py

Kept separate from the app so retraining is a deliberate, reproducible step rather
than a side effect of starting the server.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.model import AnomalyDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    settings = get_settings()
    detector = AnomalyDetector.train(
        contamination=settings.contamination,
        seed=settings.random_seed,
    )
    detector.save(settings.model_path)
    logging.info("trained and saved %s -> %s", detector.version, settings.model_path)


if __name__ == "__main__":
    main()
