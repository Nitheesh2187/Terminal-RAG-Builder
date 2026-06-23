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


def hybrid_search(
    query: str,
    *,
    k: int | None = None,
    rec: LatencyRecord | None = None,
    rerank: bool | None = None,
) -> list[Hit]:
    k = k or CFG.top_k
    rrf_k = CFG.rrf_k
    rerank_on = CFG.rerank_enabled if rerank is None else rerank

    # When reranking, pull a bigger pool so the cross-encoder has more to choose from
    pool_size = max(CFG.rerank_pool if rerank_on else k * 4, 40)
    rec = rec or LatencyRecord(command="/retrieve")

    with stage(rec, "embed_query"):
        qvec = embed_query(query).tolist()

    with connect() as conn:
        with stage(rec, "dense_sql"):
            dense_rows = dense_search(conn, qvec, pool_size)
        with stage(rec, "sparse_sql"):
            sparse_rows = sparse_search(conn, query, pool_size)

    with stage(rec, "rrf_fuse"):
        # Row layout (from db.dense_search / db.sparse_search):
        #   [0] id, [1] doc_id, [2] chunk_idx, [3] content, [4] title, [5] section,
        #   [6] score, [7] element_type
        dense_rank = {row[0]: i + 1 for i, row in enumerate(dense_rows)}
        dense_score = {row[0]: row[6] for row in dense_rows}
        sparse_rank = {row[0]: i + 1 for i, row in enumerate(sparse_rows)}
        sparse_score = {row[0]: row[6] for row in sparse_rows}

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

        # Build hits for the full RRF pool (so rerank has candidates to score)
        pool_target = CFG.rerank_pool if rerank_on else k
        candidates: list[Hit] = []
        for cid, s in fused[:pool_target]:
            row = meta[cid]
            candidates.append(Hit(
                chunk_id=row[0],
                doc_id=row[1],
                chunk_idx=row[2],
                content=row[3],
                title=row[4],
                section=row[5],
                element_type=row[7],
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
                dense_score=dense_score.get(cid),
                sparse_score=sparse_score.get(cid),
                rrf_score=s,
            ))

    if rerank_on and candidates:
        with stage(rec, "rerank"):
            from .rerank import rerank as _do_rerank
            hits = _do_rerank(query, candidates, k=k)
    else:
        hits = candidates[:k]

    rec.set("dense_hits", len(dense_rows))
    rec.set("sparse_hits", len(sparse_rows))
    rec.set("rrf_pool", len(candidates))
    rec.set("fused_returned", len(hits))
    return hits


def render_hits(hits: list[Hit], *, max_chars: int = 220) -> None:
    has_rerank = any(h.rerank_score is not None for h in hits)
    title = "hybrid retrieval (RRF + rerank)" if has_rerank else "hybrid retrieval (RRF)"
    table = Table(title=title, show_lines=False, expand=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("doc_id", style="cyan")
    table.add_column("title", style="white", max_width=26, overflow="ellipsis")
    table.add_column("section", style="blue", max_width=20, overflow="ellipsis")
    table.add_column("d_rank", justify="right", style="green")
    table.add_column("s_rank", justify="right", style="magenta")
    table.add_column("rrf", justify="right", style="yellow")
    if has_rerank:
        table.add_column("rerank", justify="right", style="bold yellow")
    table.add_column("snippet", style="dim", max_width=46, overflow="ellipsis")
    for i, h in enumerate(hits, 1):
        snippet = h.content.replace("\n", " ")[:max_chars]
        if h.element_type == "table":
            snippet = "[table] " + snippet
        row = [
            str(i),
            h.doc_id,
            h.title or "—",
            h.section or "—",
            str(h.dense_rank) if h.dense_rank else "—",
            str(h.sparse_rank) if h.sparse_rank else "—",
            f"{h.rrf_score:.4f}",
        ]
        if has_rerank:
            row.append(f"{h.rerank_score:.3f}" if h.rerank_score is not None else "—")
        row.append(snippet)
        table.add_row(*row)
    console.print(table)
