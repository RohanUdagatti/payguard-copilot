"""FastAPI application entrypoint for the PayGuard API gateway."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pythonjsonlogger.json import JsonFormatter

from payguard import __version__
from payguard.config import Settings, get_settings
from payguard.gateway.routes.ml import router as ml_router
from payguard.gateway.routes.rag import router as rag_router


def configure_logging(log_level: str) -> None:
    """Configure stdlib logging with a JSON formatter.

    Does not log request bodies or any fields that might contain PII/PAN.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level.upper())


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan hook for startup, warm-up, and shutdown logging."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    logger.info(
        "application_started",
        extra={"app_name": settings.app_name, "version": settings.app_version},
    )
    # Step 0: Warm up the embedding model at app startup
    try:
        from payguard.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(settings)
        pipeline.embedder.embed_query("warmup")
        app.state.rag_pipeline = pipeline
        logger.info("embedding_model_warmed_up")
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding_model_warmup_failed", extra={"error": str(exc)})

    # Step 1: Load fraud model if available
    try:
        from payguard.ml.service import FraudModelService

        ml_service = FraudModelService(settings)
        ml_service.load_model()
        app.state.fraud_model_service = ml_service
        logger.info(
            "fraud_model_loaded",
            extra={"model_version": ml_service.model_version},
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("fraud_model_not_loaded", extra={"reason": str(exc)})
        app.state.fraud_model_service = None

    yield
    logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application instance."""
    resolved = settings or get_settings()
    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.include_router(rag_router)
    app.include_router(ml_router)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe used by Docker and load balancers."""
        return {"status": "ok", "version": resolved.app_version or __version__}

    return app


app = create_app()
