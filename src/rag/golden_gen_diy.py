"""DIY golden dataset generator — one LLM call per question.

Much cheaper than Ragas (1 call vs 10-50 per question). 40-50 questions
fit comfortably within Groq's free-tier rate limits.

Pipeline per question:
    pick a chunk → one LLM call (JSON-mode) → validate evidence is a substring
    of the chunk (faithfulness gate) → APPEND to golden JSONL.

Source-doc selection (precedence order):
    --doc-ids    explicit comma-separated list
    --num-docs   randomly sample N pdfs from --pdf-path
    (fallback)   use the doc_ids already in the existing golden file

The output JSONL is *always* appended (created if missing — never overwrites).

Usage:
    # Sample 5 random pdfs from the configured pdf dir, generate 30 questions
    python -m rag.golden_gen_diy --n 30 --num-docs 5

    # Same, but from a custom pdf dir
    python -m rag.golden_gen_diy --n 30 --num-docs 5 --pdf-path Data/sample_pdfs

    # Specific docs
    python -m rag.golden_gen_diy --n 40 --doc-ids 2505.10468,2505.06913

    # Append more questions about whatever docs are already in the golden
    python -m rag.golden_gen_diy --n 20

    # Use the local Ollama model instead of Groq
    python -m rag.golden_gen_diy --n 30 --num-docs 5 --provider ollama

    # Override the model, and chunk source PDFs section-aware at 1200 tokens
    python -m rag.golden_gen_diy --n 30 --num-docs 5 \
        --provider ollama --model llama3.1:8b \
        --chunk-strategy section --chunk-tokens 1200

LLM backend (--provider) and chunking (--chunk-strategy / --chunk-tokens /
--chunk-overlap) default to the values in .env (LLM_PROVIDER, CHUNK_STRATEGY,
CHUNK_TOKENS, CHUNK_OVERLAP).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .chunking import chunk_sections, chunk_text, chunk_unstructured
from .config import CFG
from .pdf import pdf_to_sections, pdf_to_text, pdf_to_unstructured

console = Console()


SYSTEM_PROMPT = (
    "You write retrieval-evaluation questions grounded strictly in document "
    "excerpts. Output strict JSON only — no markdown, no commentary."
)

USER_TEMPLATE = """EXCERPT (from arXiv paper {doc_id}):
\"\"\"
{chunk_text}
\"\"\"

Write ONE question following these rules:
- It must be answerable using ONLY the excerpt above (not external knowledge).
- A researcher reading this paper might genuinely ask it.
- It must be SPECIFIC — avoid generic openers like "What is this paper about?" or "What is X?" with no context. Reference concrete entities, numbers, methods, or claims from the excerpt.
- Do NOT directly quote the excerpt verbatim in the question.
{type_hint}
Then write the answer in 1-3 sentences, grounded in the excerpt.

Also extract the EXACT substring from the excerpt that supports the answer (verbatim, character-for-character, no edits or paraphrasing).

Output strict JSON only:
{{
  "question": "...",
  "answer": "...",
  "evidence_span": "..."
}}"""


QUESTION_TYPES = [
    "Prefer a FACTOID question (asks for a specific value, name, or fact).",
    "Prefer a METHOD question (asks about a technique, approach, or procedure used).",
    "Prefer a COMPARISON question (asks how one thing differs from or compares to another).",
    "Prefer a REASONING question (asks why a result holds, or what it implies).",
    "Prefer a DEFINITION question (asks for the precise meaning of a term as used here).",
    "",  # no constraint
]


def _client(provider: str, model: str | None = None) -> tuple[OpenAI, str]:
    """Build an OpenAI-compatible client + resolved model name for `provider`.

    "groq"   → Groq cloud (needs GROQ_API_KEY).
    "ollama" → local Ollama OpenAI-compatible endpoint (no key needed).
    `model` overrides the provider's configured default when given.
    """
    provider = provider.lower()
    if provider == "ollama":
        # Ollama ignores the key but the OpenAI SDK requires a non-empty one.
        client = OpenAI(api_key="ollama", base_url=CFG.ollama_base_url)
        return client, (model or CFG.ollama_model)
    if provider == "groq":
        if not CFG.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        client = OpenAI(api_key=CFG.groq_api_key, base_url=CFG.groq_base_url)
        return client, (model or CFG.groq_model)
    raise RuntimeError(f"unknown provider {provider!r} (expected 'groq' or 'ollama')")


def _chunks_for_pdf(pdf: Path, strategy: str, max_tokens: int, overlap: int):
    """Parse + chunk a PDF with the chosen strategy; always returns list[Chunk]."""
    if strategy == "unstructured":
        sections, tables = pdf_to_unstructured(pdf)
        return chunk_unstructured(sections, tables, max_tokens=max_tokens, overlap=overlap)
    if strategy == "section":
        sections = pdf_to_sections(pdf)
        return chunk_sections(sections, max_tokens=max_tokens, overlap=overlap)
    text = pdf_to_text(pdf)
    return chunk_text(text, max_tokens=max_tokens, overlap=overlap)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _evidence_in_chunk(evidence: str, chunk: str) -> bool:
    """Faithfulness gate: evidence must appear (normalized) inside the chunk."""
    e = _normalize(evidence)
    if not e or len(e) < 8:
        return False
    return e in _normalize(chunk)


def _strip_json_fences(text: str) -> str:
    """Defensive: some models wrap JSON in ```json fences despite json mode."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t


def _generate_one(
    client: OpenAI,
    model: str,
    doc_id: str,
    chunk_text: str,
    type_hint: str,
    *,
    retries: int = 2,
) -> dict | None:
    """Single LLM call. Returns {question, answer, evidence_span} or None."""
    prompt = USER_TEMPLATE.format(
        doc_id=doc_id,
        chunk_text=chunk_text,
        type_hint=(type_hint + "\n") if type_hint else "",
    )
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
                timeout=45,
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(_strip_json_fences(raw))
            q = (data.get("question") or "").strip()
            a = (data.get("answer") or "").strip()
            ev = (data.get("evidence_span") or "").strip()
            if not (q and a and ev):
                continue
            if not _evidence_in_chunk(ev, chunk_text):
                continue
            return {"question": q, "answer": a, "evidence_span": ev}
        except json.JSONDecodeError:
            continue
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                wait = 5 * (attempt + 1)
                console.print(f"[yellow]rate-limited, sleeping {wait}s[/yellow]")
                time.sleep(wait)
            elif attempt < retries:
                time.sleep(2 ** attempt)
            else:
                console.print(f"[yellow]gen fail[/yellow]: {e}")
    return None


def _find_pdf(doc_id: str, source: Path) -> Path | None:
    """Look up a PDF for `doc_id` inside `source`. Tolerates '/' ↔ '_' encoding."""
    safe = doc_id.replace("/", "_")
    for candidate in (source / f"{safe}.pdf", source / f"{doc_id}.pdf"):
        if candidate.exists():
            return candidate
    return None


def _doc_id_from_pdf(path: Path) -> str:
    """Reverse of the '_'/'/' filename encoding used by extract.py."""
    return path.stem.replace("_", "/")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30, help="Target number of new questions.")
    p.add_argument("--num-docs", type=int, default=None,
                   help="Randomly sample this many pdfs from --pdf-path as the source. "
                        "Ignored if --doc-ids is given.")
    p.add_argument("--pdf-path", default=None,
                   help="Directory to sample pdfs from. Default: CFG.pdf_dir.")
    p.add_argument("--doc-ids", default=None,
                   help="Explicit comma-separated doc_ids. Overrides --num-docs / fallback.")
    p.add_argument("--per-doc", type=int, default=None,
                   help="Max questions per source doc. Default: ceil(n / num_docs).")
    p.add_argument("--output", default=None,
                   help="Output JSONL (always appended to, created if missing). Default: CFG.golden_path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=0.6,
                   help="Seconds between calls (rate-limit cushion).")
    p.add_argument("--provider", choices=["groq", "ollama"], default=CFG.llm_provider,
                   help="LLM backend. Default: LLM_PROVIDER env (groq).")
    p.add_argument("--model", default=None,
                   help="Model name override. Default: the provider's configured "
                        "model (GROQ_MODEL / OLLAMA_MODEL).")
    p.add_argument("--chunk-strategy", choices=["section", "recursive", "unstructured"],
                   default=CFG.chunk_strategy,
                   help="How to parse + chunk source PDFs. Default: CHUNK_STRATEGY env.")
    p.add_argument("--chunk-tokens", type=int, default=CFG.chunk_tokens,
                   help="Max tokens per chunk. Default: CHUNK_TOKENS env.")
    p.add_argument("--chunk-overlap", type=int, default=CFG.chunk_overlap,
                   help="Token overlap between chunks. Default: CHUNK_OVERLAP env.")
    args = p.parse_args()

    output = Path(args.output) if args.output else CFG.golden_path
    existing = _load_jsonl(output)
    existing_qs = {_normalize(i.get("question") or "") for i in existing}

    source_dir = Path(args.pdf_path).resolve() if args.pdf_path else CFG.pdf_dir.resolve()
    if not source_dir.exists():
        console.print(f"[red]--pdf-path does not exist[/red]: {source_dir}")
        return 1

    rng = random.Random(args.seed)

    # Resolve source doc_ids by precedence: --doc-ids > --num-docs > fallback to existing golden.
    doc_ids: list[str]
    source_label: str
    if args.doc_ids:
        doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()]
        source_label = "explicit --doc-ids"
    elif args.num_docs:
        all_pdfs = sorted(source_dir.glob("*.pdf"))
        if not all_pdfs:
            console.print(f"[red]no pdfs found in[/red] {source_dir}")
            return 1
        n_pick = min(args.num_docs, len(all_pdfs))
        if n_pick < args.num_docs:
            console.print(
                f"[yellow]note[/yellow]: --num-docs {args.num_docs} but only "
                f"{len(all_pdfs)} pdfs available — using all"
            )
        picked = rng.sample(all_pdfs, n_pick)
        doc_ids = [_doc_id_from_pdf(p) for p in picked]
        source_label = f"random sample of {n_pick} pdfs from {source_dir}"
    else:
        doc_ids = sorted({
            did for item in existing for did in (item.get("gold_doc_ids") or []) if did
        })
        source_label = f"docs already in {output.name}"
    if not doc_ids:
        console.print(
            "[red]no doc_ids to source from — pass --doc-ids, --num-docs, "
            "or have an existing golden file[/red]"
        )
        return 1

    per_doc = args.per_doc or max(1, (args.n + len(doc_ids) - 1) // len(doc_ids) + 2)
    preview = ", ".join(doc_ids[:5]) + (", ..." if len(doc_ids) > 5 else "")
    console.print(f"[cyan]source[/cyan]: {len(doc_ids)} docs ({source_label})")
    console.print(f"          {preview}")
    console.print(f"[cyan]pdf dir[/cyan]: {source_dir}")
    console.print(f"[cyan]target[/cyan]: {args.n} new questions, up to {per_doc}/doc")
    console.print(f"[cyan]existing[/cyan]: {len(existing)} questions in {output.name}")

    rng = random.Random(args.seed)
    client, model = _client(args.provider, args.model)
    console.print(f"[cyan]llm[/cyan]: {args.provider} · {model}")
    console.print(
        f"[cyan]chunking[/cyan]: {args.chunk_strategy} · {args.chunk_tokens} tok "
        f"/ {args.chunk_overlap} overlap"
    )
    new_items: list[dict] = []
    n_calls = n_fail = n_dup = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("generating", total=args.n)
        for doc_id in doc_ids:
            if len(new_items) >= args.n:
                break
            pdf = _find_pdf(doc_id, source_dir)
            if pdf is None:
                console.print(f"[yellow]no pdf for {doc_id}[/yellow]")
                continue
            try:
                chunks = _chunks_for_pdf(
                    pdf, args.chunk_strategy, args.chunk_tokens, args.chunk_overlap
                )
            except Exception as e:
                console.print(f"[yellow]parse fail {doc_id}: {e}[/yellow]")
                continue
            if not chunks:
                continue
            # Skip first chunk (often boilerplate/abstract) and last (refs/appendix)
            pool = chunks[1:-1] if len(chunks) > 3 else chunks
            sample = rng.sample(pool, min(per_doc, len(pool)))
            for ch in sample:
                if len(new_items) >= args.n:
                    break
                type_hint = rng.choice(QUESTION_TYPES)
                n_calls += 1
                res = _generate_one(client, model, doc_id, ch.text, type_hint)
                if not res:
                    n_fail += 1
                    time.sleep(args.delay)
                    continue
                if _normalize(res["question"]) in existing_qs:
                    n_dup += 1
                    continue
                existing_qs.add(_normalize(res["question"]))
                new_items.append({
                    "question": res["question"],
                    "gold_doc_ids": [doc_id],
                    "gold_answer": res["answer"],
                    "reference_contexts": [res["evidence_span"]],
                    "type": "diy",
                })
                prog.advance(task)
                time.sleep(args.delay)

    if not new_items:
        console.print(
            f"[red]no questions generated — check the {args.provider} endpoint/model, "
            "rate limits, or chunk content[/red]"
        )
        return 1

    # Always append; create dir + file if missing.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as fh:
        for item in new_items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    total = len(existing) + len(new_items)

    console.print()
    console.print("[green]done[/green]")
    console.print(f"  new questions:   {len(new_items)}")
    console.print(f"  llm calls:       {n_calls} ({n_fail} failed, {n_dup} duplicates dropped)")
    console.print(f"  final file size: {total} questions in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
