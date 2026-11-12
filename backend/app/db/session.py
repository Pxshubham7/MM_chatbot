import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from backend.app.config import get_settings
from backend.app.db.models import Base

logger = logging.getLogger(__name__)

settings = get_settings()

_engine = None
_SessionLocal = None
_active_db_type = "unknown"


def get_engine():
    """
    Initializes and returns the database engine.
    Attempts primary MySQL connection first. If unreachable and fallback is enabled,
    initializes SQLite for offline tests/development.
    """
    global _engine, _active_db_type, _SessionLocal

    if _engine is not None:
        return _engine

    primary_url = settings.database_url
    try:
        connect_args = {}
        if "mysql" in primary_url:
            connect_args["connect_timeout"] = 2

        engine = create_engine(
            primary_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args=connect_args,
            echo=False
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _engine = engine
        _active_db_type = "mysql"
        logger.info("Successfully connected to primary MySQL 8.0 database.")
    except Exception as exc:
        if settings.allow_sqlite_fallback:
            logger.warning(
                f"Primary MySQL connection failed ({exc}). Falling back to local SQLite: {settings.sqlite_db_path}"
            )
            fallback_engine = create_engine(
                settings.sqlite_db_path,
                connect_args={"check_same_thread": False},
                echo=False
            )
            _engine = fallback_engine
            _active_db_type = "sqlite"
        else:
            logger.error(f"Failed to connect to MySQL database at {primary_url}: {exc}")
            raise exc

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_db() -> Generator[Session, None, None]:
    """Dependency generator yielding an active database session."""
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_db_status() -> dict:
    """Returns the current database engine type and connectivity status."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "connected": True,
            "type": _active_db_type,
            "url": str(engine.url)
        }
    except Exception as exc:
        return {
            "connected": False,
            "type": _active_db_type,
            "error": str(exc)
        }


def init_db():
    """Idempotently creates all database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database tables initialized on {_active_db_type}.")
