"""FastAPI application entry point.

Wires together configuration, authentication, the RAG service and the
database into a runnable API. Run locally with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import rag
from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
)
from app.config import settings
from app.database import Base, engine, get_db
from app.models import Chunk, Document, QueryHistory
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentUploadResponse,
    LoginRequest,
    SourceResponse,
    TokenResponse,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_api")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload text documents and ask questions answered via RAG.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialise resources on startup.

    Creates database tables if they do not yet exist and warms up the
    ChromaDB collection. The embedding model is loaded lazily on first use.
    """
    logger.info("Starting RAG API")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ensured")
    except SQLAlchemyError as exc:
        logger.error("Failed to initialise database tables: %s", exc)
    try:
        rag.get_collection()
        logger.info("ChromaDB collection ready")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise ChromaDB: %s", exc)


# --------------------------------------------------------------------------- #
# Exception handlers
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request, exc: RequestValidationError
) -> JSONResponse:
    """Return a structured 422 response for request validation errors."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    """Catch-all handler producing a structured 500 response."""
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )


# --------------------------------------------------------------------------- #
# Health / root
# --------------------------------------------------------------------------- #
@app.get("/")
def read_root() -> dict:
    """Liveness root endpoint."""
    return {"message": "RAG API Running"}


@app.get("/health")
def health() -> dict:
    """Simple health-check endpoint."""
    return {"status": "healthy"}


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
@app.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    """Authenticate and return a JWT access token.

    Raises:
        HTTPException: ``401`` if credentials are invalid.
    """
    if not authenticate_user(payload.username, payload.password):
        logger.warning("Failed login attempt for user %s", payload.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        data={"sub": payload.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    logger.info("User %s logged in", payload.username)
    return TokenResponse(access_token=token, token_type="bearer")


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """Upload a ``.txt`` document, chunk it, embed it and persist it.

    Raises:
        HTTPException: ``400`` for invalid/empty files, ``500`` on failure.
    """
    # --- Validation --------------------------------------------------------
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are allowed",
        )

    raw = await file.read()
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 text",
        )

    # --- Processing --------------------------------------------------------
    document_id = str(uuid.uuid4())
    chunk_texts = rag.split_text(content)
    if not chunk_texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document produced no chunks",
        )

    chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]
    embeddings = rag.create_embeddings(chunk_texts)

    try:
        # Persist vectors first so we fail fast before touching the DB.
        rag.store_embeddings(
            document_id=document_id,
            filename=file.filename,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            embeddings=embeddings,
        )

        document = Document(id=uuid.UUID(document_id), filename=file.filename)
        db.add(document)
        for chunk_id, text in zip(chunk_ids, chunk_texts):
            db.add(
                Chunk(
                    id=uuid.UUID(chunk_id),
                    document_id=document.id,
                    chunk_text=text,
                )
            )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("DB error while saving document %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist document",
        )

    logger.info(
        "Uploaded document %s (%s) with %d chunks",
        document_id,
        file.filename,
        len(chunk_texts),
    )
    return DocumentUploadResponse(
        document_id=document_id, total_chunks=len(chunk_texts)
    )


# --------------------------------------------------------------------------- #
# Ask (protected)
# --------------------------------------------------------------------------- #
@app.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
) -> AskResponse:
    """Answer a question about a document using RAG. Requires a valid JWT.

    Raises:
        HTTPException: ``404`` if the document does not exist, ``500`` on
            internal failure.
    """
    # --- Verify the document exists ---------------------------------------
    try:
        document_uuid = uuid.UUID(payload.document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    document = db.query(Document).filter(Document.id == document_uuid).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    try:
        # --- Retrieve + generate ------------------------------------------
        retrieved = rag.retrieve_chunks(payload.document_id, payload.question)
        answer = rag.generate_answer(payload.question, retrieved)

        sources = [
            SourceResponse(chunk_id=chunk["chunk_id"], score=chunk["score"])
            for chunk in retrieved
        ]

        # --- Audit log -----------------------------------------------------
        db.add(
            QueryHistory(
                document_id=document.id,
                question=payload.question,
                answer=answer,
                response_length=len(answer),
            )
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("DB error while answering question: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record query",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to answer question: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer",
        )

    logger.info(
        "Answered question for document %s (user=%s)",
        payload.document_id,
        current_user,
    )
    return AskResponse(
        question=payload.question, answer=answer, sources=sources
    )
