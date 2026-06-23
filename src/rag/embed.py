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

    from . import hf_model_cached

    device = CFG.embed_device.lower()
    # local_files_only when cached → skip the per-load HEAD request to the Hub.
    return SentenceTransformer(
        CFG.embed_model, device=device, local_files_only=hf_model_cached(CFG.embed_model)
    )


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
