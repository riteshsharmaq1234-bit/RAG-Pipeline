"""Retrieval-Augmented Generation (RAG) service.

This module owns everything related to turning raw text into searchable
vectors and answering questions from them:

* :func:`split_text`        - chunk text with a sliding window.
* :func:`create_embeddings` - embed text with ``all-MiniLM-L6-v2``.
* :func:`store_embeddings`  - persist vectors + metadata in ChromaDB.
* :func:`retrieve_chunks`   - fetch the top-k chunks for a question.
* :func:`generate_answer`   - answer via OpenAI (if configured) or fallback.

The embedding model and ChromaDB client are loaded lazily and cached as
module-level singletons so the (relatively heavy) model is only loaded once.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - fallback when chromadb not installed
    chromadb = None  # type: ignore

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Tuple

from app.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
COLLECTION_NAME: str = "documents"
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_OVERLAP: int = 50
DEFAULT_TOP_K: int = 3

# Lazily-initialised singletons.
_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[object] = None
_collection: Optional[object] = None


class _SimpleCollection:
    """In-memory fallback collection used when chromadb is unavailable.

    Implements a minimal subset of the Chroma collection API used by the
    application: `add(...)` and `query(...)`. This is intentionally simple
    (not optimized) but allows the app to run without C extensions.
    """

    def __init__(self):
        self._items = []

    def add(self, ids, embeddings, documents, metadatas):
        for _id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._items.append(
                {
                    "id": _id,
                    "embedding": np.asarray(emb, dtype=float),
                    "document": doc,
                    "metadata": meta,
                }
            )

    def query(self, query_embeddings, n_results=3, where=None):
        q = np.asarray(query_embeddings[0], dtype=float)
        # filter by metadata.document_id if requested
        doc_id = (where or {}).get("document_id")
        candidates = [it for it in self._items if (doc_id is None or it["metadata"].get("document_id") == doc_id)]

        if not candidates:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

        embs = np.vstack([it["embedding"] for it in candidates])
        # cosine similarity
        q_norm = np.linalg.norm(q)
        embs_norm = np.linalg.norm(embs, axis=1)
        # avoid div by zero
        embs_norm[embs_norm == 0] = 1.0
        if q_norm == 0:
            sims = np.zeros(len(candidates))
        else:
            sims = (embs @ q) / (embs_norm * q_norm)

        # convert to distances similar to chroma (distance = 1 - similarity)
        distances = (1.0 - sims).tolist()

        # select top-k (higher similarity => lower distance)
        idx_sorted = np.argsort(distances)[:n_results]

        ids = [candidates[i]["id"] for i in idx_sorted]
        documents = [candidates[i]["document"] for i in idx_sorted]
        metadatas = [candidates[i]["metadata"] for i in idx_sorted]
        selected_distances = [distances[i] for i in idx_sorted]

        return {
            "ids": [ids],
            "documents": [documents],
            "distances": [selected_distances],
            "metadatas": [metadatas],
        }


class _SimpleClient:
    def __init__(self):
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = _SimpleCollection()
        return self._collections[name]


def get_model() -> SentenceTransformer:
    """Return the cached sentence-transformers model, loading it on demand."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection() -> "chromadb.api.models.Collection.Collection":
    """Return the persistent ChromaDB ``documents`` collection.

    The collection is created automatically on first access and configured to
    use cosine distance, which pairs well with normalised MiniLM embeddings.
    """
    global _chroma_client, _collection
    if _collection is None:
        logger.info("Initialising ChromaDB at %s", settings.CHROMA_PATH)
        if chromadb is not None:
            try:
                _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)  # type: ignore
                _collection = _chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                logger.warning("chromadb present but failed to initialise; using fallback store")
                _chroma_client = _SimpleClient()
                _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        else:
            logger.info("chromadb not installed; using in-memory fallback collection")
            _chroma_client = _SimpleClient()
            _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split ``text`` into overlapping fixed-size character chunks.

    Uses a sliding window where each step advances by
    ``chunk_size - overlap`` characters. For the defaults (500 / 50) this
    yields chunks covering 0-500, 450-950, 900-1400, and so on.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        A list of chunk strings (empty if ``text`` is empty).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text_length = len(text)
    if text_length == 0:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_length:
            break
        start += step
    return chunks


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for a list of texts.

    Args:
        texts: Input strings to embed.

    Returns:
        A list of float vectors, one per input text.
    """
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def store_embeddings(
    document_id: str,
    filename: str,
    chunk_ids: List[str],
    chunk_texts: List[str],
    embeddings: List[List[float]],
) -> None:
    """Persist chunk vectors and metadata into ChromaDB.

    Args:
        document_id: Owning document UUID (string form).
        filename: Original filename, stored as metadata.
        chunk_ids: Stable unique IDs for each chunk (used as Chroma IDs).
        chunk_texts: The chunk texts.
        embeddings: Embedding vectors aligned with ``chunk_texts``.
    """
    if not chunk_ids:
        return
    collection = get_collection()
    metadatas: List[Dict[str, str]] = [
        {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "filename": filename,
        }
        for chunk_id in chunk_ids
    ]
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
    )
    logger.info(
        "Stored %d chunks in ChromaDB for document %s",
        len(chunk_ids),
        document_id,
    )


def retrieve_chunks(
    document_id: str, question: str, top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a question within one document.

    Args:
        document_id: Restrict the search to this document.
        question: The natural-language question.
        top_k: Number of chunks to return.

    Returns:
        A list of dicts with ``chunk_id``, ``text`` and ``score`` keys,
        ordered from most to least relevant. ``score`` is a cosine
        similarity in the range [0, 1] (higher is better).
    """
    collection = get_collection()
    question_embedding = create_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where={"document_id": document_id},
    )

    # Chroma returns each field as a list-of-lists (one inner list per query).
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    chunks: List[Dict[str, Any]] = []
    for index, chunk_internal_id in enumerate(ids):
        distance = distances[index] if index < len(distances) else 1.0
        metadata = metadatas[index] if index < len(metadatas) else {}
        # Convert cosine distance -> similarity score.
        score = round(1.0 - float(distance), 4)
        chunks.append(
            {
                "chunk_id": metadata.get("chunk_id", chunk_internal_id),
                "text": documents[index] if index < len(documents) else "",
                "score": score,
            }
        )
    return chunks


def generate_answer(
    question: str, context_chunks: List[Dict[str, Any]]
) -> str:
    """Produce an answer for ``question`` grounded in ``context_chunks``.

    If ``OPENAI_API_KEY`` is configured, the OpenAI Chat Completions API is
    used with a strict context-only system prompt to reduce hallucination.
    Otherwise the text of the single most relevant chunk is returned.

    Args:
        question: The user question.
        context_chunks: Retrieved chunks (each with a ``text`` key).

    Returns:
        The generated answer string.
    """
    if not context_chunks:
        return "Information not found in document."

    context = "\n\n".join(chunk["text"] for chunk in context_chunks)

    if settings.OPENAI_API_KEY:
        try:
            # Imported lazily so the dependency is only needed when used.
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            system_prompt = (
                "Answer only using provided context. If answer is not found "
                'in context, respond: "Information not found in document."'
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.error("OpenAI call failed, falling back to chunk: %s", exc)
            return context_chunks[0]["text"]

    # No LLM configured: return the most relevant chunk verbatim.
    return context_chunks[0]["text"]
