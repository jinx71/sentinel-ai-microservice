"""Application entrypoint and factory.

The app factory pattern (create_app) lets tests build isolated instances and keeps
startup wiring in one place. The model is loaded once during the lifespan startup and
attached to app.state so every request shares the same in-memory detector.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .config import get_settings
from .metrics import MODEL_LOADED, REQUEST_COUNT, REQUEST_LATENCY
from .model import AnomalyDetector
from .routers import health, predict
from .schemas import Envelope


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logging.getLogger(__name__).info("starting %s v%s", settings.app_name, settings.app_version)

    app.state.detector = AnomalyDetector.load_or_train(
        path=settings.model_path,
        contamination=settings.contamination,
        seed=settings.random_seed,
    )
    MODEL_LOADED.set(1)
    yield
    MODEL_LOADED.set(0)
    app.state.detector = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Unsupervised anomaly detection for equipment sensor telemetry.",
        lifespan=lifespan,
    )

    # CORS is permissive here because the service is API-only; tighten the origins
    # via an env-driven allowlist if a browser client is ever added.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observe_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        # Use the route template (not the raw path) to keep label cardinality bounded.
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
        REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
        return response

    app.include_router(health.router)
    app.include_router(predict.router)

    @app.get("/", tags=["meta"])
    async def root() -> Envelope[dict]:
        return Envelope(
            success=True,
            data={"service": settings.app_name, "version": settings.app_version},
            message="ok",
        )

    @app.get("/metrics", tags=["meta"])
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
