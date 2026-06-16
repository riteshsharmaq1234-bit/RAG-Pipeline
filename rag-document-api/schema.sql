-- ===========================================================================
-- RAG Document Q&A API - PostgreSQL Schema
-- ===========================================================================
-- Requires PostgreSQL 13+ for gen_random_uuid() (built-in via pgcrypto).
-- If on an older version, run:  CREATE EXTENSION IF NOT EXISTS "pgcrypto";
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- Table: documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents (filename);

-- ---------------------------------------------------------------------------
-- Table: chunks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_text  TEXT NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);

-- ---------------------------------------------------------------------------
-- Table: query_history
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    response_length INTEGER NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_history_document_id ON query_history (document_id);
CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history (created_at);

-- ===========================================================================
-- Analytics Queries
-- ===========================================================================

-- SQL Query #1: Most Queried Documents
SELECT d.filename,
       COUNT(*) AS query_count
FROM query_history q
JOIN documents d ON q.document_id = d.id
GROUP BY d.filename
ORDER BY query_count DESC;

-- SQL Query #2: Average Query Response Length Per Day
SELECT DATE(created_at) AS day,
       AVG(response_length) AS avg_length
FROM query_history
GROUP BY DATE(created_at);
