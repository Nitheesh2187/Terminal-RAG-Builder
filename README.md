# Terminal RAG Builder

A terminal-first RAG over arXiv papers. Hybrid search (dense pgvector + sparse Postgres FTS, fused with RRF), latency tables on every command, and a golden-set evaluation harness.

## Stack
- **DB**: Postgres 16 + pgvector (HNSW cosine) + native `tsvector` for sparse, via docker-compose
- **Embeddings**: `BAAI/bge-small-en-v1.5` (384-dim, local, CPU-fast)
- **LLM**: Groq (OpenAI-compatible) — `llama-3.3-70b-versatile` by default
- **CLI**: `prompt_toolkit` REPL + `rich` tables

## One-time setup

```bash
# 1. PDFs (uses the extractor we already built)
python3 extract.py --count 5000

# 2. Postgres
docker compose up -d

# 3. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. Config
cp .env.example .env
# edit .env and paste your free key from https://console.groq.com
```

## Run

```bash
rag           # or: python -m rag.cli
```

You'll get a REPL:

```
rag> /ingest --limit 5000          # parse PDFs → chunk → embed → upsert
rag> /retrieve attention mechanism transformers -k 10
rag> What are the key ideas of transformer attention?   # full RAG (retrieve + generate)
rag> /evaluate --gen -k 10         # run golden set, print metrics + latencies
rag> /stats
rag> /exit
```

Every command prints a latency table with per-stage timings (ms, % of total) and a counters table.

## Pipelines

### `/ingest`
PDF parse → chunking → batched embeddings → upsert into `documents` + `chunks`. Stage timings: `pdf_parse`, `chunk`, `embed`, `upsert`. Counters: docs/sec, chunks/sec, tokens indexed.

Parsing + chunking are selected by `CHUNK_STRATEGY`:
- `section` — PyMuPDF, TOC/heading-aware section chunking.
- `recursive` — PyMuPDF flat text, recursive token-aware splitter (800 toks / 100 overlap).
- `unstructured` — layout-aware parse via the [`unstructured`](https://github.com/Unstructured-IO/unstructured) library. Tables are extracted as their own elements and chunked **separately** (`element_type='table'`, no overlap, never folded into prose); narrative text is section-chunked as usual. Needs `pip install 'unstructured[pdf]'`. The default `hi_res` strategy (required for table-structure inference) also needs the system binaries **`poppler`** and **`tesseract-ocr`** — `sudo apt install poppler-utils tesseract-ocr`. Tune with `UNSTRUCTURED_STRATEGY` (`hi_res` for tables, `fast` for text-only / no system deps) and `UNSTRUCTURED_INFER_TABLES`.

### `/retrieve` and free-form query
1. Embed query (with bge query prefix).
2. Dense ANN: `embedding <=> qvec` (HNSW cosine).
3. Sparse: `ts_rank(tsv, plainto_tsquery(...))`.
4. **RRF fusion**: `score = Σ 1 / (k_rrf + rank_i)`, k_rrf=60. Both ranked lists pulled at 4× target k.
5. (For free-form queries) Groq generation with `[#]` citations.

### `/evaluate`
Reads JSONL golden file:
```json
{"question": "...", "gold_doc_ids": ["0704.0003"], "gold_answer": null}
```
Per query: runs full pipeline and computes Hit@1, Recall@k, MRR@k, nDCG@k. Aggregates per-stage P50/P95/P99 latencies. LLM-judge metrics (faithfulness, answer-relevancy) are stubbed for the next phase, once `gold_answer` is populated.

Default golden path: `src/rag/golden/golden.jsonl` (see `golden.example.jsonl`).

## Layout
```
src/rag/
  cli.py            REPL + command dispatch
  config.py         env-driven config
  db.py             pgvector schema + pool
  pdf.py            PDF parsing (PyMuPDF sections + unstructured layout/tables)
  chunking.py       token-aware splitter (+ table-isolating unstructured chunker)
  embed.py          sentence-transformers wrapper
  ingest.py         /ingest pipeline
  retrieve.py       /retrieve pipeline (hybrid + RRF)
  generate.py       Groq answer with citations
  evaluate.py       /evaluate pipeline (golden runner + metrics)
  metrics.py        Timer + Rich latency tables
  golden/           place golden.jsonl here
```

## What's next (deliberately deferred)
- Cross-encoder reranker stage
- LLM-as-judge for faithfulness/answer-relevancy
- HNSW tuning, query rewriting / multi-query
- Per-section / sentence-window chunking for academic PDFs
