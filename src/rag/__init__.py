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

__version__ = "0.1.0"
