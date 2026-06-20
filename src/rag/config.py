"""Centralised, env-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    database_url: str = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    embed_device: str = os.getenv("EMBED_DEVICE", "cpu")
    embed_batch: int = int(os.getenv("EMBED_BATCH", "64"))

    pdf_dir: Path = ROOT / os.getenv("PDF_DIR", "pdfs")
    metadata_path: Path = ROOT / os.getenv("METADATA_PATH", "metadata.json")
    golden_path: Path = ROOT / os.getenv("GOLDEN_PATH", "src/rag/golden/golden.jsonl")

    top_k: int = int(os.getenv("TOP_K", "10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))


CFG = Config()
