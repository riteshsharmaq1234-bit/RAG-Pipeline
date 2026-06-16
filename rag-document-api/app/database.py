"""Database layer.

Defines the SQLAlchemy engine, session factory, declarative ``Base`` and a
FastAPI dependency (``get_db``) that yields a request-scoped session and
guarantees it is closed afterwards.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import settings

# ``pool_pre_ping`` transparently recovers from stale/dropped connections,
# which is important for long-running production services.
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

# Declarative base shared by all ORM models.
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session.

    Yields:
        A SQLAlchemy ``Session`` that is automatically closed when the
        request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
