"""Inference endpoints: batch prediction and single-reading explanation."""

import time

from fastapi import APIRouter, HTTPException, Request, status

from ..config import get_settings
from ..explain import llm_explanation
from ..metrics import ANOMALIES_TOTAL, PREDICTION_LATENCY, PREDICTIONS_TOTAL
from ..schemas import (
    Envelope,
    ExplainData,
    ExplainRequest,
    PredictData,
    PredictionResult,
    PredictRequest,
)

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/predict", response_model=Envelope[PredictData])
async def predict(payload: PredictRequest, request: Request) -> Envelope[PredictData]:
    """Score a batch of sensor readings for anomalies."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )

    readings = [r.model_dump() for r in payload.readings]
    start = time.perf_counter()
    try:
        scored = detector.predict(readings)
    except Exception as exc:  # noqa: BLE001 - surface as 500 with a clean envelope
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="inference failed",
        ) from exc
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    anomalies = sum(1 for r in scored if r["is_anomaly"])
    PREDICTIONS_TOTAL.inc(len(scored))
    ANOMALIES_TOTAL.inc(anomalies)

    data = PredictData(
        model_version=detector.version,
        count=len(scored),
        anomalies=anomalies,
        results=[PredictionResult(**r) for r in scored],
    )
    return Envelope(success=True, data=data, message="scored")


@router.post("/explain", response_model=Envelope[ExplainData])
async def explain(payload: ExplainRequest, request: Request) -> Envelope[ExplainData]:
    """Score one reading and return a plain-language explanation.

    Uses the LLM only when configured; otherwise returns the deterministic
    rule-based narrative. Either way the response shape is identical.
    """
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )

    reading = payload.reading.model_dump()
    result = detector.predict([reading])[0]
    text, source = llm_explanation(reading, result, get_settings())

    data = ExplainData(
        equipment_id=result["equipment_id"],
        is_anomaly=result["is_anomaly"],
        severity=result["severity"],
        anomaly_score=result["anomaly_score"],
        explanation=text,
        source=source,
    )
    return Envelope(success=True, data=data, message="explained")
