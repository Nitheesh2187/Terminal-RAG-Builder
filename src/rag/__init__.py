"""Terminal-friendly RAG over arXiv PDFs."""
import os as _os
from pathlib import Path as _Path

# Load .env FIRST — before any submodule imports torch — so EMBED_DEVICE
# set in .env actually controls device selection. Errors here are silent;
# config.py loads .env again later as the authoritative pass.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

# Hide GPUs from PyTorch unless the user opts in via EMBED_DEVICE=cuda.
# Must happen BEFORE torch is imported anywhere, because torch probes CUDA
# at import time and an old/broken NVIDIA driver crashes that probe.
# If the user asked for CUDA, actively *remove* any lingering empty
# CUDA_VISIBLE_DEVICES from the shell — otherwise torch sees zero devices.
if _os.environ.get("EMBED_DEVICE", "cpu").lower() == "cuda":
    if _os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        del _os.environ["CUDA_VISIBLE_DEVICES"]
else:
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Per-model offline detection. sentence-transformers makes a metadata HEAD
# request to huggingface.co on EVERY model load — slow at best, fatal when
# offline — so we pass local_files_only=True to those loads once the model is
# cached (see embed.py / rerank.py). We deliberately do NOT set a *global*
# HF_HUB_OFFLINE: that would also block the "unstructured" strategy from lazily
# downloading its layout/table models from the Hub on first use (and can't see a
# --chunk-strategy chosen at runtime, after this module is imported).
_HF_HOME = _Path(_os.environ.get("HF_HOME", _Path.home() / ".cache" / "huggingface"))


def hf_model_cached(model_id: str) -> bool:
    """True if `model_id` is already present in the local HF hub cache.

    Pass the result as local_files_only= to a sentence-transformers load to skip
    the network HEAD check when the model is on disk, while still allowing a
    download on the first (cache-miss) run.
    """
    return (_HF_HOME / "hub" / f"models--{model_id.replace('/', '--')}").exists()


__version__ = "0.1.0"
