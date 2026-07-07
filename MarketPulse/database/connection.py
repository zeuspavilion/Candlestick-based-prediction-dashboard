import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from MarketPulse.config import DATABASE_URL

logger = logging.getLogger("marketpulse.database")

# Setup connection arguments based on DB dialect
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    # Avoid "database is locked" errors in SQLite by adding timeout
    connect_args = {"check_same_thread": False, "timeout": 30}

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,  # Test connections before using them
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db_session = scoped_session(SessionLocal)
except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine: {e}")
    raise e

Base = declarative_base()

def get_db():
    """FastAPI dependency to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
