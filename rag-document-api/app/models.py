"""SQLAlchemy ORM models.

Three tables back the service:

* ``documents``      - one row per uploaded file.
* ``chunks``         - the text chunks produced from a document.
* ``query_history``  - an audit log of every question asked.

All tables use UUID primary keys and server-friendly timestamps. Foreign
keys cascade on delete so removing a document cleans up its dependents.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Document(Base):
    """An uploaded source document."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False, index=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    queries = relationship(
        "QueryHistory",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Document id={self.id} filename={self.filename!r}>"


class Chunk(Base):
    """A single text chunk derived from a :class:`Document`."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Chunk id={self.id} document_id={self.document_id}>"


class QueryHistory(Base):
    """An audit record of a question/answer interaction."""

    __tablename__ = "query_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    response_length = Column(Integer, nullable=False)
    created_at = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    document = relationship("Document", back_populates="queries")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<QueryHistory id={self.id} document_id={self.document_id}>"
