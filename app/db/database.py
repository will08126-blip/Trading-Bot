"""
Database engine and session management.
Uses SQLite via SQLAlchemy async.
"""
import os
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import get_settings


class Base(DeclarativeBase):
    pass


def _get_db_url() -> str:
    settings = get_settings()
    url = settings.database_url
    # Convert sync sqlite:/// to async aiosqlite:///
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        # Ensure directory exists
        db_dir = os.path.dirname(os.path.abspath(path))
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{path}"
    return url


engine = create_async_engine(
    _get_db_url(),
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in get_settings().database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables if they don't exist."""
    # Import all models to register them with Base
    import app.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
