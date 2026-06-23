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

# Flip HuggingFace to offline mode once every model we'll load is cached on
# disk. Without this, sentence-transformers makes a metadata HEAD request to
# huggingface.co on EVERY model load — slow at best, fatal when offline.
# First-time runs still hit the network (cache miss), then go offline forever.
_hf_home = _Path(_os.environ.get("HF_HOME", _Path.home() / ".cache" / "huggingface"))
_required_models = [_os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")]
if _os.environ.get("RERANK_ENABLED", "false").lower() in ("1", "true", "yes"):
    _required_models.append(_os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base"))
_all_cached = all(
    (_hf_home / "hub" / f"models--{m.replace('/', '--')}").exists()
    for m in _required_models
)
if _all_cached:
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.1.0"
