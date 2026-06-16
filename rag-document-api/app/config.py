"""Application configuration.

Loads all environment-driven settings from a ``.env`` file (or the real
environment) using ``python-dotenv``. A single ``settings`` instance is
imported across the application so configuration is read once at startup.
"""

from __future__ import annotations
from pydantic import BaseSettings

import os
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file if present. Real environment
# variables always take precedence over .env values.
load_dotenv()


class Settings(BaseSettings):
    """Strongly-typed accessor for environment configuration.

    Attributes:
        DATABASE_URL: SQLAlchemy connection string for PostgreSQL.
        JWT_SECRET_KEY: Secret used to sign JWT access tokens.
        JWT_ALGORITHM: Signing algorithm (HS256 by default).
        ACCESS_TOKEN_EXPIRE_MINUTES: Token lifetime in minutes.
        OPENAI_API_KEY: Optional OpenAI key; enables LLM-backed answers.
        CHROMA_PATH: Local on-disk path for the ChromaDB persistent store.
    """

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/rag_db",
    )
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY",
        "CHANGE_ME_IN_PRODUCTION_use_a_long_random_string",
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    # An empty string in the environment is treated as "not set".
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or None
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")


# Single shared settings instance.
settings = Settings()
