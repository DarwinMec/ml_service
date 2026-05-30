from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings


settings = get_settings()

engine: Engine = create_engine(
    settings.ml_db_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_engine() -> Engine:
    """
    Retorna el engine global de SQLAlchemy.
    """
    return engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager para operaciones con sesión SQLAlchemy.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """
    Ejecuta una consulta SQL y retorna un DataFrame.
    """
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection, params=params)


def execute_sql(query: str, params: dict | None = None) -> None:
    """
    Ejecuta una instrucción SQL sin retorno.
    """
    with engine.begin() as connection:
        connection.execute(text(query), params or {})


def test_connection() -> bool:
    """
    Verifica si la conexión a PostgreSQL está funcionando.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False