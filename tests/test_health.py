"""Tests for the health endpoint and configuration loader."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from payguard.config import Settings, get_settings
from payguard.gateway.main import create_app


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Ensure Settings cache does not leak state across tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a FastAPI test client bound to a fresh app instance."""
    settings = Settings(
        _env_file=None,
        app_name="payguard-test",
        app_version="0.1.0-test",
        log_level="WARNING",
    )
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    """GET /health returns 200 with status and version."""
    response = client.get("/health")
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0-test"


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings reads values from environment variables."""
    monkeypatch.setenv("APP_NAME", "payguard-env")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_USER", "pguser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pgsecret")
    monkeypatch.setenv("POSTGRES_DB", "pgdb")
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "redissecret")
    monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-key")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.internal:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    settings = Settings(_env_file=None)

    assert settings.app_name == "payguard-env"
    assert settings.app_version == "9.9.9"
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
    assert settings.postgres_host == "db.internal"
    assert settings.postgres_port == 5433
    assert settings.postgres_user == "pguser"
    assert settings.postgres_password == "pgsecret"
    assert settings.postgres_db == "pgdb"
    assert settings.redis_host == "redis.internal"
    assert settings.redis_port == 6380
    assert settings.redis_password == "redissecret"
    assert settings.qdrant_host == "qdrant.internal"
    assert settings.qdrant_port == 6334
    assert settings.qdrant_api_key == "qdrant-key"
    assert settings.langfuse_host == "http://langfuse.internal:3000"
    assert settings.langfuse_public_key == "pk-test"
    assert settings.langfuse_secret_key == "sk-test"
    assert "pguser:pgsecret@db.internal:5433/pgdb" in settings.postgres_dsn
    assert settings.redis_url == "redis://:redissecret@redis.internal:6380/0"
    assert settings.qdrant_url == "http://qdrant.internal:6334"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings returns a cached instance until the cache is cleared."""
    monkeypatch.setenv("APP_NAME", "cached-app")
    first = get_settings()
    monkeypatch.setenv("APP_NAME", "other-app")
    second = get_settings()
    assert first is second
    assert first.app_name == "cached-app"
    # Ensure env pollution from this test does not affect others.
    assert "APP_NAME" in os.environ
