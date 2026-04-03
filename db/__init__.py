from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from .models import Base


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve a database URL from explicit value or environment variables."""
    if database_url:
        return database_url
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URI")
        or "postgresql+psycopg2://postgres:postgres@localhost:5432/f1tenth_genesis"
    )


def create_db_engine(database_url: str | None = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine using the configured database URL."""
    resolved_url = resolve_database_url(database_url)
    return create_engine(
        resolved_url,
        echo=echo,
        pool_pre_ping=True,
        future=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the default session factory used by app code."""
    return sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )


def bootstrap_database(
    database_url: str | None = None,
    *,
    echo: bool = False,
) -> tuple[Engine, sessionmaker[Session]]:
    """Create the engine, ensure all tables exist, and return a session factory."""
    engine = create_db_engine(database_url=database_url, echo=echo)
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    return engine, session_factory
