"""Cross-encoder reranker.

Reranks the candidate pool from hybrid_search by encoding (query, chunk)
pairs together with a cross-encoder, which is more discriminating than the
bi-encoder embeddings used for initial retrieval.

Loaded lazily on first use; reused thereafter.
"""
from __future__ import annotations

import os
from functools import lru_cache

from .config import CFG
from .models import Hit


@lru_cache(maxsize=1)
def _reranker():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import CrossEncoder

    device = os.getenv("EMBED_DEVICE", "cpu")
    return CrossEncoder(CFG.rerank_model, device=device, max_length=512)


def rerank(query: str, hits: list[Hit], k: int) -> list[Hit]:
    """Re-score hits via cross-encoder; mutates rerank_score, returns top-k."""
    if not hits:
        return hits
    pairs = [(query, h.content) for h in hits]
    scores = _reranker().predict(pairs, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    hits_sorted = sorted(hits, key=lambda h: h.rerank_score or float("-inf"), reverse=True)
    return hits_sorted[:k]
