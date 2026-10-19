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
    """Application lifespan hook for startup and shutdown logging."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    logger.info(
        "application_started",
        extra={"app_name": settings.app_name, "version": settings.app_version},
    )
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

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe used by Docker and load balancers."""
        return {"status": "ok", "version": resolved.app_version or __version__}

    return app


app = create_app()
