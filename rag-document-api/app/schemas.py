"""Pydantic v2 request/response schemas.

These models define the public contract of the API and provide automatic
validation and OpenAPI documentation. ``from_attributes=True`` allows the
response models to be built directly from ORM objects when convenient.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    """Credentials submitted to ``POST /login``."""

    username: str = Field(..., examples=["admin"])
    password: str = Field(..., examples=["admin123"])


class TokenResponse(BaseModel):
    """JWT access token returned on successful login."""

    access_token: str
    token_type: str = "bearer"


class AskRequest(BaseModel):
    """Body for ``POST /ask``."""

    document_id: str = Field(..., description="UUID of the target document.")
    question: str = Field(..., min_length=1, description="User question.")


class SourceResponse(BaseModel):
    """A single retrieved chunk cited as a source for the answer."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    score: float


class AskResponse(BaseModel):
    """Answer payload returned by ``POST /ask``."""

    model_config = ConfigDict(from_attributes=True)

    question: str
    answer: str
    sources: List[SourceResponse]


class DocumentUploadResponse(BaseModel):
    """Result of a successful ``POST /documents`` upload."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    total_chunks: int
