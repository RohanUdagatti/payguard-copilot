"""Database connection and session management for PayGuard."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from payguard.config import get_settings

logger = logging.getLogger("payguard.data.db")


def get_db_url(url: str | None = None) -> str:
    """Resolve database URL ensuring psycopg driver prefix."""
    raw_url = url or get_settings().postgres_dsn
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw_url


def create_db_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Create an Engine with connection pooling and timeouts."""
    db_url = get_db_url(url)
    return create_engine(
        db_url,
        echo=echo,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the global or initialized database engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_db_engine()
    return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    """Return the global session factory."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SESSION_FACTORY


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database transactional sessions."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose and reset the cached engine and session factory."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        _ENGINE.dispose()
        _ENGINE = None
    _SESSION_FACTORY = None

