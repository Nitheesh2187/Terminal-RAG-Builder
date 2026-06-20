"""Local dense embeddings via sentence-transformers (singleton-loaded)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import CFG


@lru_cache(maxsize=1)
def _model():
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    device = CFG.embed_device.lower()
    return SentenceTransformer(CFG.embed_model, device=device)


def embed_texts(texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
    """Return L2-normalized float32 embeddings, shape (len(texts), embed_dim)."""
    if not texts:
        return np.zeros((0, CFG.embed_dim), dtype=np.float32)
    vecs = _model().encode(
        texts,
        batch_size=batch_size or CFG.embed_batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """bge-* models prepend a query instruction for asymmetric search."""
    prompt = text
    if "bge" in CFG.embed_model.lower():
        prompt = "Represent this sentence for searching relevant passages: " + text
    return embed_texts([prompt])[0]
