"""Postgres + pgvector: connection pool, schema, and all DB queries."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import CFG
from .models import Chunk

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

"""

INDEX_SQL = """

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


# ---------------------------------------------------------------------------
# Document + chunk write paths (used by ingest)
# ---------------------------------------------------------------------------

def existing_doc_ids() -> set[str]:
    """All doc_ids currently in the documents table."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM documents;")
        return {r[0] for r in cur.fetchall()}


def upsert_document(conn, doc_id: str, pdf_path: str, meta: dict, n_chunks: int) -> None:
    """Insert or update a document row by doc_id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (doc_id, title, authors, categories, year, abstract, pdf_path, n_chunks)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE
              SET title = EXCLUDED.title,
                  authors = EXCLUDED.authors,
                  categories = EXCLUDED.categories,
                  year = EXCLUDED.year,
                  abstract = EXCLUDED.abstract,
                  pdf_path = EXCLUDED.pdf_path,
                  n_chunks = EXCLUDED.n_chunks,
                  ingested_at = NOW();
            """,
            (
                doc_id,
                meta.get("title"),
                meta.get("authors"),
                meta.get("categories"),
                meta.get("year"),
                meta.get("abstract"),
                pdf_path,
                n_chunks,
            ),
        )


def insert_chunks(conn, doc_id: str, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
    """Replace all chunks for doc_id with the given (chunks, embeddings)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s;", (doc_id,))
        rows = [
            (doc_id, c.idx, c.text, c.n_tokens, embeddings[i])
            for i, c in enumerate(chunks)
        ]
        cur.executemany(
            "INSERT INTO chunks (doc_id, chunk_idx, content, n_tokens, embedding)"
            " VALUES (%s, %s, %s, %s, %s);",
            rows,
        )


# ---------------------------------------------------------------------------
# Read paths (used by retrieve)
# ---------------------------------------------------------------------------

def dense_search(conn, qvec, k: int) -> list[tuple]:
    """Top-k chunks by cosine similarity to qvec, joined with document title.

    Returns rows of: (id, doc_id, chunk_idx, content, title, score).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.chunk_idx, c.content, d.title,
                   1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            LEFT JOIN documents d ON d.doc_id = c.doc_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
            """,
            (qvec, qvec, k),
        )
        return cur.fetchall()


def sparse_search(conn, query: str, k: int) -> list[tuple]:
    """Top-k chunks by Postgres FTS ts_rank, joined with document title."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.chunk_idx, c.content, d.title,
                   ts_rank(c.tsv, plainto_tsquery('english', %s)) AS score
            FROM chunks c
            LEFT JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s;
            """,
            (query, query, k),
        )
        return cur.fetchall()
