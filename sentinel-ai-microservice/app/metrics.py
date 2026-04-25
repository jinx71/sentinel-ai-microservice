"""Prometheus metrics.

Why: exposing /metrics turns the service into something an MLOps stack can observe.
We track request volume/latency (operational health) and model behaviour (anomaly
rate, prediction latency) so data drift shows up as a shift in the anomaly ratio.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total individual sensor readings scored",
)

ANOMALIES_TOTAL = Counter(
    "anomalies_detected_total",
    "Total readings flagged as anomalous",
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time spent scoring one batch",
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the anomaly model is loaded and ready, else 0",
)
