"""Bootstrap our golden file from the public `dwb2023/ragas-golden-dataset`.

The dataset has ~12 Ragas-generated Q/A pairs about three arXiv papers:
    arXiv:2505.10468, arXiv:2505.06913, arXiv:2505.06817

This script:
  1. Downloads the HF dataset
  2. Inspects its columns and maps each row to our schema
  3. Per-row doc_id assignment:
       - if a column directly identifies the source paper → use it
       - else substring-match the contexts against the three arxiv IDs
       - else assign all three arxiv IDs (loose match) and warn
  4. Downloads the three PDFs into PDF_DIR if missing
  5. Updates metadata.json with their titles (best effort from arXiv API)
  6. Writes our golden.jsonl

Run:
    python -m rag.ragas_golden_setup
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console

from .config import CFG

console = Console()

DATASET_REPO = "dwb2023/ragas-golden-dataset"
ARXIV_IDS = ["2505.10468", "2505.06913", "2505.06817"]
USER_AGENT = "Terminal-RAG-Builder/1.0 (mailto:nitheesh.raju@skylarklabs.ai)"


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------

def load_dataset_rows() -> list[dict]:
    """Pull every split's rows out of dwb2023/ragas-golden-dataset."""
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from datasets import load_dataset

    console.print(f"[cyan]loading[/cyan] {DATASET_REPO}...")
    ds = load_dataset(DATASET_REPO)
    rows: list[dict] = []
    for split in ds:
        for r in ds[split]:
            rows.append(dict(r))
    if not rows:
        raise RuntimeError("dataset returned zero rows")
    console.print(f"  {len(rows)} rows ({len(ds)} split(s))")
    console.print(f"  columns: {sorted(rows[0].keys())}")
    return rows


# ---------------------------------------------------------------------------
# column mapping — figure out which row field has what
# ---------------------------------------------------------------------------

QUESTION_KEYS = ("user_input", "question", "query")
ANSWER_KEYS   = ("reference", "ground_truth", "answer", "gold_answer", "response")
CONTEXT_KEYS  = ("reference_contexts", "contexts", "retrieved_contexts")
SOURCE_KEYS   = ("doc_id", "arxiv_id", "paper_id", "source", "filename", "file_name")


def _first_present(row: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row:
            return k
    return None


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(v) for v in value)
    return str(value)


def _extract_doc_ids_from_text(text: str, candidates: list[str]) -> list[str]:
    """Return the arXiv ids from `candidates` that appear in `text`."""
    hits: list[str] = []
    for aid in candidates:
        if aid in text:
            hits.append(aid)
    return hits


# ---------------------------------------------------------------------------
# arxiv metadata + PDF download
# ---------------------------------------------------------------------------

def fetch_arxiv_meta(arxiv_id: str) -> dict:
    """Best-effort fetch of title/abstract via the arXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        console.print(f"[yellow]meta fetch failed[/yellow] for {arxiv_id}: {e}")
        return {"id": arxiv_id}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(body)
        entry = root.find("a:entry", ns)
        if entry is None:
            return {"id": arxiv_id}
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        authors = [a.findtext("a:name", default="", namespaces=ns)
                   for a in entry.findall("a:author", ns)]
        return {
            "id": arxiv_id,
            "title": re.sub(r"\s+", " ", title),
            "abstract": re.sub(r"\s+", " ", summary),
            "authors": ", ".join(filter(None, authors)) or None,
        }
    except Exception:
        return {"id": arxiv_id}


def download_pdf(arxiv_id: str, dest_dir: Path, delay_seconds: float = 0.0) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = arxiv_id.replace("/", "_")
    out = dest_dir / f"{safe}.pdf"
    if out.exists() and out.stat().st_size > 0:
        console.print(f"  {arxiv_id} — cached")
        return True
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        out.write_bytes(data)
        console.print(f"  {arxiv_id} — OK ({len(data) // 1024} KB)")
        if delay_seconds:
            time.sleep(delay_seconds)
        return True
    except (urllib.error.URLError, TimeoutError) as e:
        console.print(f"  {arxiv_id} — [red]FAIL[/red]: {e}")
        return False


# ---------------------------------------------------------------------------
# metadata + golden writers
# ---------------------------------------------------------------------------

def update_metadata_json(papers: list[dict]) -> None:
    p = CFG.metadata_path
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            data = {"samples": []}
    else:
        data = {"samples": []}
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = {s["id"] for s in data.get("samples", [])}
    added = 0
    for meta in papers:
        if meta["id"] in existing:
            continue
        data.setdefault("samples", []).append({
            "id": meta["id"],
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "categories": None,
            "year": meta["id"][:2] and ("20" + meta["id"][:2]) or None,
            "abstract": meta.get("abstract"),
            "source": "ragas-golden",
        })
        added += 1
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    console.print(f"[green]metadata[/green]: appended {added} rows to {p}")


def write_golden(rows: list[dict], output: Path) -> int:
    qkey = _first_present(rows[0], QUESTION_KEYS)
    akey = _first_present(rows[0], ANSWER_KEYS)
    ckey = _first_present(rows[0], CONTEXT_KEYS)
    skey = _first_present(rows[0], SOURCE_KEYS)
    console.print(
        f"[cyan]column map[/cyan]: question={qkey}, answer={akey}, contexts={ckey}, source={skey}"
    )
    if qkey is None:
        raise RuntimeError(f"no question column found in {sorted(rows[0].keys())}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    unmapped = 0
    with output.open("w") as fh:
        for row in rows:
            question = _stringify(row.get(qkey)).strip()
            if not question:
                continue
            answer = _stringify(row.get(akey)).strip() if akey else None

            # Preserve the source's reference contexts as-is (list of strings).
            ref_contexts: list[str] = []
            if ckey:
                raw_ctx = row.get(ckey)
                if isinstance(raw_ctx, list):
                    ref_contexts = [str(c) for c in raw_ctx if c]
                elif raw_ctx:
                    ref_contexts = [_stringify(raw_ctx)]

            # Per-row doc-id resolution
            gold_ids: list[str] = []
            if skey and row.get(skey):
                raw = _stringify(row[skey])
                gold_ids = _extract_doc_ids_from_text(raw, ARXIV_IDS)
            if not gold_ids and ref_contexts:
                gold_ids = _extract_doc_ids_from_text(
                    "\n".join(ref_contexts), ARXIV_IDS
                )
            if not gold_ids:
                gold_ids = _extract_doc_ids_from_text(
                    f"{question}\n{answer or ''}", ARXIV_IDS
                )
            if not gold_ids:
                gold_ids = list(ARXIV_IDS)
                unmapped += 1

            item = {
                "question": question,
                "gold_doc_ids": gold_ids,
                "gold_answer": answer or None,
                "reference_contexts": ref_contexts,
                "type": "ragas-golden",
            }
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            written += 1

    console.print(f"[green]golden[/green]: wrote {written} questions to {output}")
    if unmapped:
        console.print(
            f"[yellow]note[/yellow]: {unmapped} row(s) had no detectable arxiv id "
            f"in source/contexts/text — assigned all 3 as gold (loose match)"
        )
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=None, help="Golden JSONL path. Default: CFG.golden_path.")
    p.add_argument("--pdf-dir", default=None, help="PDF dir. Default: CFG.pdf_dir.")
    p.add_argument("--delay", type=float, default=3.0, help="Delay between PDF downloads (arxiv asks ~3s).")
    args = p.parse_args()

    output = Path(args.output) if args.output else CFG.golden_path
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else CFG.pdf_dir

    rows = load_dataset_rows()
    written = write_golden(rows, output)

    console.print(f"[cyan]downloading[/cyan] PDFs for {ARXIV_IDS}...")
    metas = []
    ok = 0
    for aid in ARXIV_IDS:
        meta = fetch_arxiv_meta(aid)
        metas.append(meta)
        if download_pdf(aid, pdf_dir, delay_seconds=args.delay):
            ok += 1

    update_metadata_json(metas)

    console.print()
    console.print("[green]done[/green]")
    console.print(f"  questions in golden: {written}")
    console.print(f"  pdfs ready:          {ok}/{len(ARXIV_IDS)}")
    console.print(f"  next: [bold]rag → /ingest[/bold] then [bold]/evaluate --ragas -k 10[/bold]")
    return 0 if ok == len(ARXIV_IDS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
