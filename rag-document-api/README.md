# RAG Document Q&A API

A production-style **Retrieval-Augmented Generation (RAG)** service built with
**FastAPI**. Upload `.txt` documents, have them chunked and embedded with
`sentence-transformers` (`all-MiniLM-L6-v2`), stored as vectors in **ChromaDB**
with metadata in **PostgreSQL**, and ask natural-language questions that are
answered from the retrieved context — optionally via the **OpenAI** Chat
Completions API, with a deterministic offline fallback. Protected by **JWT**
authentication.

---

## 1. Architecture Diagram

```
                         ┌──────────────────────────────┐
                         │            Client             │
                         │   (curl / Swagger / app)      │
                         └───────────────┬──────────────┘
                                         │ HTTPS (JSON / multipart)
                                         ▼
                         ┌──────────────────────────────┐
                         │           FastAPI             │
                         │  /login  /documents  /ask     │
                         │  /  /health                   │
                         │  ── JWT auth (OAuth2 Bearer)  │
                         └───────┬───────────────┬──────┘
                                 │               │
              embeddings +       │               │  metadata + audit
              vector search      │               │
                                 ▼               ▼
              ┌─────────────────────────┐  ┌──────────────────────────┐
              │        ChromaDB         │  │        PostgreSQL         │
              │  collection: documents  │  │  documents / chunks /     │
              │  (persistent, on disk)  │  │  query_history            │
              └─────────────────────────┘  └──────────────────────────┘
                                 ▲
                                 │ encode()
                    ┌─────────────────────────────┐
                    │     sentence-transformers     │
                    │       all-MiniLM-L6-v2        │
                    └─────────────────────────────┘
                                 │ (optional)
                                 ▼
                    ┌─────────────────────────────┐
                    │   OpenAI Chat Completions     │
                    │     (LLM answer synthesis)    │
                    └─────────────────────────────┘
```

---

## 2. Request Flow

**Upload (`POST /documents`)**

1. Validate the file is a non-empty `.txt`.
2. Read and decode the content (UTF-8).
3. Generate a document UUID.
4. Split the text into overlapping chunks.
5. Embed each chunk with `all-MiniLM-L6-v2`.
6. Store vectors + metadata in ChromaDB (collection `documents`).
7. Persist the document and its chunks in PostgreSQL.
8. Return `{ document_id, total_chunks }` with `201 Created`.

**Ask (`POST /ask`, requires JWT)**

1. Validate the JWT (401 if missing/invalid/expired).
2. Verify the document exists (404 otherwise).
3. Embed the question.
4. Query ChromaDB for the top 3 chunks filtered by `document_id`.
5. Build the retrieval context.
6. Generate the answer (OpenAI if configured, otherwise the top chunk).
7. Log the interaction in `query_history`.
8. Return `{ question, answer, sources[] }` with `200 OK`.

---

## 3. Chunking Strategy

A **sliding-window** strategy over characters:

- `chunk_size = 500`
- `overlap = 50`
- step = `chunk_size - overlap = 450`

So chunks cover `0–500`, `450–950`, `900–1400`, and so on until the document
ends. The 50-character overlap keeps sentences that straddle a boundary intact
in at least one chunk, which improves retrieval recall. The final chunk is
emitted once and never duplicated as an empty trailing slice.

---

## 4. Why ChromaDB

- **Zero-ops local persistence** — `PersistentClient` writes to disk
  (`./chroma_db`), so there is no separate vector-DB server to run for
  development.
- **Metadata filtering** — native `where={"document_id": ...}` filtering lets
  us scope retrieval to a single document cheaply.
- **Cosine similarity** — configured with `hnsw:space = cosine`, which pairs
  naturally with the normalised MiniLM embeddings.
- **Simple API** — `add()` / `query()` map directly onto the store/retrieve
  steps, keeping the RAG layer small and readable.
- **Easy upgrade path** — the same interface can later point at a hosted
  Chroma server without changing call sites.

---

## 5. Hallucination Reduction Strategy

- **Context-only prompting** — the system prompt instructs the model to answer
  *only* from the supplied context and to reply
  `"Information not found in document."` when the answer is absent.
- **Tight retrieval scope** — answers are grounded in the top 3 chunks of a
  single document, never the whole corpus.
- **`temperature = 0`** — deterministic, conservative generation.
- **Source citations** — every answer returns the `chunk_id` and similarity
  `score` of its supporting chunks so answers are auditable.
- **Deterministic fallback** — without an OpenAI key, the API returns the most
  relevant chunk verbatim instead of inventing prose.

---

## 6. PostgreSQL Schema

`documents`

| column      | type         | notes               |
|-------------|--------------|---------------------|
| id          | UUID         | primary key         |
| filename    | VARCHAR(255) | indexed             |
| uploaded_at | TIMESTAMP    | default now         |

`chunks`

| column      | type      | notes                                  |
|-------------|-----------|----------------------------------------|
| id          | UUID      | primary key                            |
| document_id | UUID      | FK → documents(id), ON DELETE CASCADE  |
| chunk_text  | TEXT      |                                        |
| created_at  | TIMESTAMP | default now                            |

`query_history`

| column          | type      | notes                                  |
|-----------------|-----------|----------------------------------------|
| id              | UUID      | primary key                            |
| document_id     | UUID      | FK → documents(id), ON DELETE CASCADE  |
| question        | TEXT      |                                        |
| answer          | TEXT      |                                        |
| response_length | INTEGER   |                                        |
| created_at      | TIMESTAMP | default now, indexed                   |

See `schema.sql` for the full DDL plus the two analytics queries (most queried
documents; average response length per day).

---

## 7. Environment Variables

| Variable                      | Description                                  | Default                          |
|-------------------------------|----------------------------------------------|----------------------------------|
| `DATABASE_URL`                | PostgreSQL connection string                 | `postgresql://postgres:password@localhost:5432/rag_db` |
| `JWT_SECRET_KEY`              | Secret used to sign JWTs                      | *(change in production)*         |
| `JWT_ALGORITHM`               | JWT signing algorithm                         | `HS256`                          |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime in minutes                     | `60`                             |
| `OPENAI_API_KEY`              | Optional; enables LLM-backed answers          | *(empty → offline fallback)*     |
| `CHROMA_PATH`                 | On-disk path for the Chroma store             | `./chroma_db`                    |

Copy `.env.example` to `.env` and edit the values.

---

## 8. Setup Instructions For Windows

The steps below assume Windows 10/11 with PowerShell. (On macOS/Linux the
Python steps are identical; only the activation command differs.)

1. Install **Python 3.10+** from <https://www.python.org/downloads/> — tick
   *"Add Python to PATH"* during installation.
2. Install **PostgreSQL** (see section 9).
3. Clone or unzip this project and open a terminal in the project root.
4. Follow sections 10–13.

---

## 9. PostgreSQL Installation

1. Download the installer from
   <https://www.postgresql.org/download/windows/> (EnterpriseDB build).
2. Run the installer; set a password for the `postgres` superuser and keep the
   default port `5432`.
3. After install, open **SQL Shell (psql)** or **pgAdmin** and create the
   database:

   ```sql
   CREATE DATABASE rag_db;
   ```

4. Update `DATABASE_URL` in your `.env` with the correct user/password/host.

---

## 10. Virtual Environment Setup

From the project root:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 11. Dependency Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` model (~80 MB) from Hugging
> Face and caches it locally, so the initial document upload may take a little
> longer.

---

## 12. Database Setup

Apply the schema (either is fine — the app also auto-creates tables on
startup):

**Using psql:**

```bash
psql -U postgres -d rag_db -f schema.sql
```

**Using pgAdmin:** open `schema.sql` in the Query Tool and run it against
`rag_db`.

---

## 13. Running Application

```bash
uvicorn app.main:app --reload
```

- API root: <http://127.0.0.1:8000/>
- Interactive docs (Swagger UI): <http://127.0.0.1:8000/docs>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>

---

## 14. API Examples

**Login**

```
POST /login
{ "username": "admin", "password": "admin123" }
→ { "access_token": "<jwt>", "token_type": "bearer" }
```

**Upload a document**

```
POST /documents   (multipart/form-data, field: file=<your.txt>)
→ 201 { "document_id": "…", "total_chunks": 15 }
```

**Ask a question**

```
POST /ask   (Authorization: Bearer <jwt>)
{ "document_id": "…", "question": "What is the refund policy?" }
→ 200 {
    "question": "What is the refund policy?",
    "answer": "…",
    "sources": [ { "chunk_id": "…", "score": 0.93 } ]
  }
```

---

## 15. Sample cURL Commands

```bash
# 1) Login and capture the token
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# 2) Upload a .txt document
curl -s -X POST http://127.0.0.1:8000/documents \
  -F "file=@sample.txt"

# 3) Ask a question (use the document_id returned above)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id":"<DOCUMENT_ID>","question":"What is this document about?"}'
```

On Windows PowerShell, replace `$TOKEN` handling with `Invoke-RestMethod` or
paste the token manually.

---

## 16. Future Improvements

- **User store + roles** — replace the hardcoded admin with a real users table
  and refresh tokens.
- **Token-aware chunking** — chunk on sentence/token boundaries instead of raw
  characters for cleaner context windows.
- **Re-ranking** — add a cross-encoder re-ranker over the top-k results.
- **Streaming answers** — stream OpenAI responses to the client.
- **More formats** — support PDF, DOCX and Markdown ingestion.
- **Async embeddings + background jobs** — offload large uploads to a queue.
- **Observability** — structured JSON logs, request IDs, and metrics.
- **Containerisation** — Docker Compose for API + PostgreSQL + Chroma.
- **Tests** — unit tests for chunking/auth and integration tests for the API.

---

## Project Structure

```
rag-document-api/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app, routes, startup, exception handlers
│   ├── auth.py        # JWT auth (login, token, dependencies)
│   ├── rag.py         # chunking, embeddings, Chroma, answer generation
│   ├── database.py    # engine, session, Base, get_db dependency
│   ├── schemas.py     # Pydantic v2 request/response models
│   ├── models.py      # SQLAlchemy ORM models
│   └── config.py      # environment configuration
├── schema.sql         # PostgreSQL DDL + analytics queries
├── requirements.txt   # pinned dependencies
├── .env.example       # environment template
├── .gitignore
└── README.md
```

## Notes

- **Default credentials:** `admin` / `admin123` (change before any real use).
- `POST /ask` is JWT-protected; `POST /documents` is open by default — add the
  `get_current_user` dependency to protect it if required.
