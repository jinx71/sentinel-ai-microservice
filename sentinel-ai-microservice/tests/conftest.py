"""Shared test fixtures.

LLM explanations are forced off and a temp model path is used so the suite is fully
hermetic — no network, no API key, no artifact left in the repo.
"""

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    model_path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    os.environ["ENABLE_LLM_EXPLANATIONS"] = "false"
    os.environ["MODEL_PATH"] = str(model_path)

    # Import after env is set so get_settings() picks it up on first read.
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:  # triggers lifespan startup -> model load
        yield test_client


# A clearly-nominal reading and a clearly-anomalous one, used across tests.
NORMAL_READING = {
    "equipment_id": "WFI-PUMP-03",
    "temperature": 22.0,
    "pressure": 4.0,
    "vibration": 2.5,
    "humidity": 45.0,
    "flow_rate": 120.0,
}

ANOMALOUS_READING = {
    "equipment_id": "WFI-PUMP-09",
    "temperature": 65.0,
    "pressure": 12.0,
    "vibration": 28.0,
    "humidity": 92.0,
    "flow_rate": 15.0,
}
