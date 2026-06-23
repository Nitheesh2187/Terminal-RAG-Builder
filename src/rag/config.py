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

    # Local Ollama (OpenAI-compatible endpoint). Default LLM backend is "groq";
    # set LLM_PROVIDER=ollama (or pass --provider ollama) to use the local model.
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

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
    # "section"      = TOC-aware section chunking (pdf_to_sections + chunk_sections)
    # "recursive"    = flat text + recursive token splitter
    # "unstructured" = layout/table-aware parse via the `unstructured` library;
    #                  tables are isolated and chunked separately (chunk_unstructured)
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive").lower()

    # Knobs for CHUNK_STRATEGY="unstructured". "hi_res" is required for table
    # structure inference (HTML); "fast" is text-only and far lighter but loses
    # tables. hi_res additionally needs system 'poppler' + 'tesseract-ocr'.
    unstructured_strategy: str = os.getenv("UNSTRUCTURED_STRATEGY", "hi_res").lower()
    unstructured_infer_tables: bool = os.getenv("UNSTRUCTURED_INFER_TABLES", "true").lower() in ("1", "true", "yes")

    # Cross-encoder reranker (optional). When enabled, hybrid_search pulls a
    # bigger candidate pool from RRF, then re-scores each (query, chunk) pair
    # with a cross-encoder and returns the top-k by that score.
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    rerank_pool: int = int(os.getenv("RERANK_POOL", "30"))

    # No-LLM context precision/recall (n-gram overlap between retrieved chunks
    # and gold reference spans). Tune the threshold with the per-span soft
    # scores printed by /evaluate — see metrics.context_overlap_metrics.
    ctx_overlap_n: int = int(os.getenv("CTX_OVERLAP_N", "3"))
    ctx_overlap_threshold: float = float(os.getenv("CTX_OVERLAP_THRESHOLD", "0.45"))


CFG = Config()
