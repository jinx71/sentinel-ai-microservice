"""Request and response schemas.

Why: every API endpoint returns a consistent {success, data, message} envelope so
clients can branch on one shape regardless of which route they hit.
"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["normal", "low", "medium", "high"]


class SensorReading(BaseModel):
    """A single equipment telemetry sample.

    Field bounds reflect plausible cleanroom / utility ranges; values outside them
    are rejected at the edge (422) before ever reaching the model.
    """

    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(..., min_length=1, max_length=64, examples=["WFI-PUMP-03"])
    temperature: float = Field(..., ge=-50.0, le=200.0, description="degrees Celsius")
    pressure: float = Field(..., ge=0.0, le=50.0, description="bar")
    vibration: float = Field(..., ge=0.0, le=100.0, description="mm/s RMS")
    humidity: float = Field(..., ge=0.0, le=100.0, description="percent RH")
    flow_rate: float = Field(..., ge=0.0, le=500.0, description="litres/min")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readings: list[SensorReading] = Field(..., min_length=1, max_length=500)


class PredictionResult(BaseModel):
    equipment_id: str
    anomaly_score: float = Field(..., description="higher = more anomalous")
    is_anomaly: bool
    severity: Severity


class PredictData(BaseModel):
    model_version: str
    count: int
    anomalies: int
    results: list[PredictionResult]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reading: SensorReading


class ExplainData(BaseModel):
    equipment_id: str
    is_anomaly: bool
    severity: Severity
    anomaly_score: float
    explanation: str
    source: Literal["rule_based", "llm"]


T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    message: str = ""
