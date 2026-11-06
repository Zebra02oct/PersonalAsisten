"""
Database Connection Module

MySQL connection using SQLAlchemy ORM.
Manages connection pooling and session handling.
"""

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get database credentials from environment
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'asisten_db')

# Create connection URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy Base for models
Base = declarative_base()


class DatabaseConfig:
    """Database configuration and connection manager."""

    _engine = None
    _session_factory = None

    @classmethod
    def get_engine(cls):
        """Get or create database engine."""
        if cls._engine is None:
            cls._engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connection before use
                echo=False  # Set to True for SQL debugging
            )
        return cls._engine

    @classmethod
    def get_session_factory(cls):
        """Get or create session factory."""
        if cls._session_factory is None:
            engine = cls.get_engine()
            cls._session_factory = sessionmaker(
                bind=engine,
                autocommit=False,
                autoflush=False
            )
        return cls._session_factory

    @classmethod
    def get_session(cls) -> Generator[Session, None, None]:
        """
        Get database session (context manager).

        Usage:
            with Database.get_session() as session:
                session.query(User).all()
        """
        session_factory = cls.get_session_factory()
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def create_tables(cls):
        """Create all tables in the database."""
        from . import models  # Import models to register them
        engine = cls.get_engine()
        Base.metadata.create_all(engine)

    @classmethod
    def drop_tables(cls):
        """Drop all tables (USE WITH CAUTION!)."""
        Base.metadata.drop_all(cls.get_engine())

    @classmethod
    def close(cls):
        """Close all connections."""
        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
            cls._session_factory = None


# Decorator for database transactions
def with_session(func):
    """Decorator to provide database session to functions."""
    def wrapper(*args, **kwargs):
        with DatabaseConfig.get_session() as session:
            return func(*args, session=session, **kwargs)
    return wrapper


# Test connection
def test_connection() -> bool:
    """Test database connection."""
    try:
        engine = DatabaseConfig.get_engine()
        with engine.connect() as conn:
            conn.ping()
        print("[OK] Database connection successful")
        return True
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()
