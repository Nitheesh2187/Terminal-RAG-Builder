"""Postgres + pgvector connection pool and schema management."""
from __future__ import annotations

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import CFG

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT,
    authors     TEXT,
    categories  TEXT,
    year        TEXT,
    abstract    TEXT,
    pdf_path    TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_idx   INTEGER NOT NULL,
    content     TEXT NOT NULL,
    n_tokens    INTEGER NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (doc_id, chunk_idx)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
"""

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def _ensure_extension() -> None:
    """Create the vector extension via a plain connection so the pool's
    configure() can safely register the vector type on every new connection."""
    with psycopg.connect(CFG.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _ensure_extension()
        _pool = ConnectionPool(
            CFG.database_url,
            min_size=1,
            max_size=8,
            configure=_configure,
            kwargs={"autocommit": False},
        )
        _pool.wait()
    return _pool


@contextmanager
def connect():
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def init_schema() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.format(dim=CFG.embed_dim))
        conn.commit()


def reset_schema() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
        conn.commit()
    init_schema()


def counts() -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents;")
        ndocs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks;")
        nchunks = cur.fetchone()[0]
    return {"documents": ndocs, "chunks": nchunks}
