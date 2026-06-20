"""PDF → text → chunks → embeddings → pgvector upsert, with per-stage timings."""
from __future__ import annotations

import json
import time
from pathlib import Path

import fitz  # PyMuPDF
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .chunking import chunk_text
from .config import CFG
from .db import connect, counts, init_schema, reset_schema
from .embed import embed_texts
from .metrics import LatencyRecord, render_latency, stage

console = Console()


def _load_metadata_index() -> dict[str, dict]:
    """Map arxiv_id → minimal metadata from metadata.json if present (optional)."""
    p = CFG.metadata_path
    if not p.exists():
        console.print(f"[dim]no metadata file at {p} — ingesting without titles/abstracts[/dim]")
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        console.print(f"[yellow]metadata file at {p} is not valid JSON — skipping[/yellow]")
        return {}
    out: dict[str, dict] = {}
    for s in data.get("samples", []):
        out[s["id"]] = {
            "title": s.get("title"),
            "authors": s.get("authors"),
            "categories": s.get("categories"),
            "year": s.get("year"),
            "abstract": s.get("abstract"),
        }
    return out


def _pdf_to_text(path: Path) -> str:
    with fitz.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def _existing_doc_ids() -> set[str]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM documents;")
        return {r[0] for r in cur.fetchall()}


def _upsert_document(conn, doc_id: str, pdf_path: str, meta: dict, n_chunks: int) -> None:
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


def _insert_chunks(conn, doc_id: str, chunks, embeddings) -> None:
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


def run_ingest(*, limit: int | None = None, reset: bool = False) -> LatencyRecord:
    rec = LatencyRecord(command="/ingest")
    if reset:
        with stage(rec, "reset_schema"):
            reset_schema()
    else:
        with stage(rec, "init_schema"):
            init_schema()

    meta_index = _load_metadata_index()
    pdf_paths = sorted(CFG.pdf_dir.glob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    skip = _existing_doc_ids() if not reset else set()
    pending = [p for p in pdf_paths if p.stem.replace("_", "/") not in skip and p.stem not in skip]

    console.print(
        f"[cyan]ingest[/cyan]: {len(pdf_paths)} pdfs found, "
        f"{len(pending)} pending, {len(pdf_paths) - len(pending)} already ingested"
    )
    if not pending:
        rec.set("documents_total", counts()["documents"])
        rec.set("chunks_total", counts()["chunks"])
        render_latency(rec)
        return rec

    parse_ms = chunk_ms = embed_ms = upsert_ms = 0.0
    n_chunks_total = 0
    n_tokens_total = 0
    n_docs_ok = 0
    n_docs_fail = 0
    started = time.perf_counter()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("ingesting", total=len(pending))
        with connect() as conn:
            for pdf in pending:
                doc_id = pdf.stem.replace("_", "/")
                meta = meta_index.get(doc_id, {})
                try:
                    t0 = time.perf_counter()
                    text = _pdf_to_text(pdf)
                    parse_ms += (time.perf_counter() - t0) * 1000

                    t0 = time.perf_counter()
                    chunks = chunk_text(
                        text, max_tokens=CFG.chunk_tokens, overlap=CFG.chunk_overlap
                    )
                    chunk_ms += (time.perf_counter() - t0) * 1000
                    if not chunks:
                        n_docs_fail += 1
                        prog.advance(task)
                        continue

                    t0 = time.perf_counter()
                    vecs = embed_texts([c.text for c in chunks])
                    embed_ms += (time.perf_counter() - t0) * 1000

                    t0 = time.perf_counter()
                    _upsert_document(conn, doc_id, str(pdf.relative_to(CFG.root)), meta, len(chunks))
                    _insert_chunks(conn, doc_id, chunks, vecs)
                    conn.commit()
                    upsert_ms += (time.perf_counter() - t0) * 1000

                    n_docs_ok += 1
                    n_chunks_total += len(chunks)
                    n_tokens_total += sum(c.n_tokens for c in chunks)
                except Exception as e:
                    conn.rollback()
                    console.print(f"[red]fail[/red] {doc_id}: {e}")
                    n_docs_fail += 1
                prog.advance(task)

    total_s = time.perf_counter() - started
    rec.add("pdf_parse", parse_ms)
    rec.add("chunk", chunk_ms)
    rec.add("embed", embed_ms)
    rec.add("upsert", upsert_ms)
    rec.set("docs_ingested", n_docs_ok)
    rec.set("docs_failed", n_docs_fail)
    rec.set("chunks_created", n_chunks_total)
    rec.set("tokens_indexed", n_tokens_total)
    rec.set("docs_per_sec", n_docs_ok / total_s if total_s > 0 else 0.0)
    rec.set("chunks_per_sec", n_chunks_total / total_s if total_s > 0 else 0.0)
    c = counts()
    rec.set("documents_total", c["documents"])
    rec.set("chunks_total", c["chunks"])
    render_latency(rec)
    return rec
