"""Hybrid retrieval: dense (pgvector cosine) + sparse (tsvector ts_rank) fused via RRF."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .config import CFG
from .db import connect, dense_search, sparse_search
from .embed import embed_query
from .metrics import stage
from .models import Hit, LatencyRecord

console = Console()


def hybrid_search(query: str, *, k: int | None = None, rec: LatencyRecord | None = None) -> list[Hit]:
    k = k or CFG.top_k
    rrf_k = CFG.rrf_k
    pool_size = max(k * 4, 40)
    rec = rec or LatencyRecord(command="/retrieve")

    with stage(rec, "embed_query"):
        qvec = embed_query(query).tolist()

    with connect() as conn:
        with stage(rec, "dense_sql"):
            dense_rows = dense_search(conn, qvec, pool_size)
        with stage(rec, "sparse_sql"):
            sparse_rows = sparse_search(conn, query, pool_size)

    with stage(rec, "rrf_fuse"):
        dense_rank = {row[0]: i + 1 for i, row in enumerate(dense_rows)}
        dense_score = {row[0]: row[5] for row in dense_rows}
        sparse_rank = {row[0]: i + 1 for i, row in enumerate(sparse_rows)}
        sparse_score = {row[0]: row[5] for row in sparse_rows}

        meta = {row[0]: row for row in dense_rows}
        for row in sparse_rows:
            meta.setdefault(row[0], row)

        fused: list[tuple[int, float]] = []
        for cid in meta:
            s = 0.0
            if cid in dense_rank:
                s += 1.0 / (rrf_k + dense_rank[cid])
            if cid in sparse_rank:
                s += 1.0 / (rrf_k + sparse_rank[cid])
            fused.append((cid, s))
        fused.sort(key=lambda x: x[1], reverse=True)

        hits: list[Hit] = []
        for cid, s in fused[:k]:
            row = meta[cid]
            hits.append(Hit(
                chunk_id=row[0],
                doc_id=row[1],
                chunk_idx=row[2],
                content=row[3],
                title=row[4],
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
                dense_score=dense_score.get(cid),
                sparse_score=sparse_score.get(cid),
                rrf_score=s,
            ))

    rec.set("dense_hits", len(dense_rows))
    rec.set("sparse_hits", len(sparse_rows))
    rec.set("fused_returned", len(hits))
    return hits


def render_hits(hits: list[Hit], *, max_chars: int = 220) -> None:
    table = Table(title="hybrid retrieval (RRF)", show_lines=False, expand=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("doc_id", style="cyan")
    table.add_column("title", style="white", max_width=40, overflow="ellipsis")
    table.add_column("d_rank", justify="right", style="green")
    table.add_column("s_rank", justify="right", style="magenta")
    table.add_column("rrf", justify="right", style="yellow")
    table.add_column("snippet", style="dim", max_width=60, overflow="ellipsis")
    for i, h in enumerate(hits, 1):
        snippet = h.content.replace("\n", " ")[:max_chars]
        table.add_row(
            str(i),
            h.doc_id,
            h.title or "—",
            str(h.dense_rank) if h.dense_rank else "—",
            str(h.sparse_rank) if h.sparse_rank else "—",
            f"{h.rrf_score:.4f}",
            snippet,
        )
    console.print(table)
