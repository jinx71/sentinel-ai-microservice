"""Liveness and readiness endpoints.

Why two separate probes: liveness answers "is the process alive?" (restart if not),
while readiness answers "can it serve traffic yet?" (the model must be loaded).
Kubernetes uses them differently — conflating them causes pods to be killed during
slow model loads or to receive traffic before they are ready.
"""

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def liveness() -> dict:
    """Always 200 while the event loop is responsive."""
    return {"status": "alive"}


@router.get("/readyz")
async def readiness(request: Request, response: Response) -> dict:
    """200 only once the model is attached to app state."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "model not loaded"}
    return {"status": "ready", "model_version": detector.version}
