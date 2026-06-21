"""Generate a synthetic golden dataset using Ragas' TestsetGenerator.

Pipeline:
    sample N pdfs → parse → chunk → wrap each chunk as a LangChain Document
    with metadata{doc_id} → TestsetGenerator → map contexts back to doc_ids
    → write JSONL in our golden format.

Run as a module:
    python -m rag.golden_gen --count 100 --output Data/golden/golden.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from rich.console import Console

from .chunking import chunk_text
from .config import CFG
from .metrics import build_ragas_embeddings, build_ragas_llm
from .pdf import pdf_to_text

console = Console()


def _load_metadata_index() -> dict:
    p = CFG.metadata_path
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {s["id"]: s for s in data.get("samples", [])}


def _sample_pdfs(n: int, seed: int) -> list[Path]:
    pdfs = sorted(CFG.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"no PDFs in {CFG.pdf_dir}")
    rng = random.Random(seed)
    return rng.sample(pdfs, min(n, len(pdfs)))


def _build_documents(pdf_paths: list[Path], chunks_per_doc: int, seed: int):
    """Parse each PDF, chunk it, sample chunks, wrap as LangChain Documents."""
    from langchain.schema import Document  # local import: only needed for generation

    rng = random.Random(seed + 1)
    meta_idx = _load_metadata_index()
    docs = []
    for p in pdf_paths:
        doc_id = p.stem.replace("_", "/")
        try:
            text = pdf_to_text(p)
        except Exception as e:
            console.print(f"[yellow]skip {doc_id}: {e}[/yellow]")
            continue
        chunks = chunk_text(text, max_tokens=CFG.chunk_tokens, overlap=CFG.chunk_overlap)
        if not chunks:
            continue
        sample = (
            chunks if chunks_per_doc >= len(chunks)
            else rng.sample(chunks, chunks_per_doc)
        )
        title = (meta_idx.get(doc_id, {}).get("title") or "").strip()
        for c in sample:
            docs.append(Document(
                page_content=c.text,
                metadata={"doc_id": doc_id, "chunk_idx": c.idx, "title": title},
            ))
    return docs


def _map_contexts_to_doc_ids(context_text: str, docs) -> list[str]:
    """Substring-match a Ragas context back to source docs to recover doc_id(s)."""
    ctx_norm = " ".join((context_text or "").split())
    if not ctx_norm:
        return []
    head = ctx_norm[:200]
    seen, out = set(), []
    for d in docs:
        body = " ".join(d.page_content.split())
        if head and head in body:
            did = d.metadata.get("doc_id")
            if did and did not in seen:
                seen.add(did)
                out.append(did)
    return out


def generate_golden(
    *,
    count: int,
    output: Path,
    seed: int = 42,
    chunks_per_doc: int = 3,
    pdf_sample: int | None = None,
) -> Path:
    """Generate `count` synthetic Q/A items and write to `output` (JSONL)."""
    from ragas.testset import TestsetGenerator

    n_pdfs = pdf_sample or max(count, 50)
    console.print(
        f"[cyan]golden_gen[/cyan]: target {count} questions, "
        f"sampling {n_pdfs} pdfs, {chunks_per_doc} chunks/doc, seed={seed}"
    )

    pdf_paths = _sample_pdfs(n_pdfs, seed)
    docs = _build_documents(pdf_paths, chunks_per_doc, seed)
    if not docs:
        raise RuntimeError("no documents built — check pdf_dir and parsing")
    console.print(f"  built {len(docs)} chunk-docs from {len(pdf_paths)} pdfs")

    llm = build_ragas_llm()
    embed = build_ragas_embeddings()
    gen = TestsetGenerator(llm=llm, embedding_model=embed)

    console.print(
        "[cyan]generating[/cyan] — Ragas will make many LLM calls "
        "(this is the slow part on the free Groq tier)"
    )
    t0 = time.perf_counter()
    testset = gen.generate_with_langchain_docs(docs, testset_size=count)
    df = testset.to_pandas()
    console.print(f"  done in {time.perf_counter() - t0:.1f}s, {len(df)} rows")

    # Ragas 0.2+ schema: user_input / reference / reference_contexts / synthesizer_name
    qcol = "user_input" if "user_input" in df.columns else "question"
    acol = "reference" if "reference" in df.columns else "ground_truth"
    ccol = (
        "reference_contexts" if "reference_contexts" in df.columns
        else "contexts" if "contexts" in df.columns else None
    )
    tcol = "synthesizer_name" if "synthesizer_name" in df.columns else None

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w") as fh:
        for _, row in df.iterrows():
            contexts = row.get(ccol) if ccol else []
            if isinstance(contexts, str):
                contexts = [contexts]
            elif contexts is None:
                contexts = []
            doc_ids: list[str] = []
            for c in contexts:
                doc_ids.extend(_map_contexts_to_doc_ids(c, docs))
            seen, gold = set(), []
            for d in doc_ids:
                if d not in seen:
                    seen.add(d); gold.append(d)
            item: dict = {
                "question": row.get(qcol),
                "gold_doc_ids": gold,
                "gold_answer": row.get(acol),
            }
            if tcol:
                item["type"] = row.get(tcol)
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            written += 1
    console.print(f"[green]wrote[/green] {written} items to {output}")
    unmapped = sum(1 for _ in [None] * 0)  # placeholder; we already log per-write
    # quick sanity: how many ended up with empty gold_doc_ids
    with output.open() as fh:
        empties = sum(1 for line in fh if not json.loads(line).get("gold_doc_ids"))
    if empties:
        console.print(
            f"[yellow]warn[/yellow]: {empties}/{written} items had no doc_id mapped "
            f"(contexts didn't substring-match any source chunk)"
        )
    return output


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", "-n", type=int, default=50, help="Number of Q/A to generate.")
    p.add_argument("--output", "-o", default=None, help="Output JSONL path. Default: CFG.golden_path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunks-per-doc", type=int, default=3)
    p.add_argument("--pdf-sample", type=int, default=None,
                   help="Number of PDFs to sample from (default: max(count, 50)).")
    args = p.parse_args()
    out = Path(args.output) if args.output else CFG.golden_path
    generate_golden(
        count=args.count, output=out, seed=args.seed,
        chunks_per_doc=args.chunks_per_doc, pdf_sample=args.pdf_sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
