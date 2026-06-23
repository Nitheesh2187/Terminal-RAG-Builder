"""Build a smaller PDF directory for fast ingest-eval experiments.

Always includes every PDF referenced by the golden file (so /evaluate can
actually score them); fills the remaining slots with a reproducible random
sample from the full corpus. Symlinks by default — no disk duplication.

Workflow:
    python -m rag.sample_pdfs --count 200
    # edit .env: PDF_DIR=Data/pdfs_sample
    rag> /ingest --reset
    rag> /evaluate

To switch back to the full corpus: revert PDF_DIR in .env, /ingest --reset.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from rich.console import Console

from .config import CFG

console = Console()


def load_golden_doc_ids(path: Path) -> set[str]:
    """Read gold_doc_ids from a golden JSONL (one record per line)."""
    if not path.exists():
        console.print(f"[yellow]no golden file at {path} — no docs will be pinned[/yellow]")
        return set()
    ids: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for did in rec.get("gold_doc_ids") or []:
                if did:
                    ids.add(str(did))
    return ids


def _index_pdfs(source: Path) -> dict[str, Path]:
    """Map both '/' and '_' arxiv-id forms to their PDF file in source.

    Our PDFs are saved with '_' substituted for '/' (e.g. 'cs/0501001' →
    'cs_0501001.pdf'). Accept either form as a key so callers don't have to
    care which one is in the golden file.
    """
    out: dict[str, Path] = {}
    for p in sorted(source.glob("*.pdf")):
        stem = p.stem
        out.setdefault(stem.replace("_", "/"), p)
        out.setdefault(stem, p)
    return out


def _wipe_dir(d: Path) -> None:
    if not d.exists():
        return
    for f in d.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", "-n", type=int, default=200,
                   help="Total docs in the sample (golden docs + random fill).")
    p.add_argument("--source", default=None,
                   help="Source PDF dir. Default: CFG.pdf_dir.")
    p.add_argument("--output", default="Data/sample_pdfs",
                   help="Output dir for symlinks/copies.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlinking (uses disk).")
    p.add_argument("--reset", action="store_true",
                   help="Wipe the output dir before populating it.")
    args = p.parse_args()

    source = Path(args.source).resolve() if args.source else CFG.pdf_dir.resolve()
    output = (CFG.root / args.output).resolve() if not Path(args.output).is_absolute() \
             else Path(args.output).resolve()

    if not source.exists():
        console.print(f"[red]source dir not found[/red]: {source}")
        return 1

    all_pdfs = sorted(source.glob("*.pdf"))
    if not all_pdfs:
        console.print(f"[red]no PDFs in[/red] {source}")
        return 1
    console.print(f"[cyan]corpus[/cyan]: {len(all_pdfs)} pdfs in {source}")

    pdfs_by_id = _index_pdfs(source)

    golden_ids = load_golden_doc_ids(CFG.golden_path)
    console.print(f"[cyan]golden[/cyan]: {len(golden_ids)} doc ids referenced")

    golden_paths: list[Path] = []
    missing: list[str] = []
    seen_paths: set[Path] = set()
    for gid in sorted(golden_ids):
        pdf = pdfs_by_id.get(gid)
        if pdf is None:
            missing.append(gid)
            continue
        rp = pdf.resolve()
        if rp not in seen_paths:
            seen_paths.add(rp)
            golden_paths.append(pdf)
    if missing:
        sample = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        console.print(f"[yellow]warn[/yellow]: {len(missing)} golden id(s) have no PDF in source ({sample})")

    if len(golden_paths) > args.count:
        console.print(
            f"[yellow]note[/yellow]: golden has {len(golden_paths)} docs but --count is {args.count}; "
            f"using all golden, skipping random fill"
        )
        chosen = list(golden_paths)
    else:
        pool = [p for p in all_pdfs if p.resolve() not in seen_paths]
        rng = random.Random(args.seed)
        need = args.count - len(golden_paths)
        fill = rng.sample(pool, min(need, len(pool)))
        chosen = list(golden_paths) + fill

    # Materialise the output dir
    if args.reset:
        _wipe_dir(output)
    output.mkdir(parents=True, exist_ok=True)

    n_link, n_copy, n_skip, n_fail = 0, 0, 0, 0
    for src in chosen:
        dest = output / src.name
        if dest.exists() or dest.is_symlink():
            n_skip += 1
            continue
        try:
            if args.copy:
                shutil.copy2(src, dest)
                n_copy += 1
            else:
                dest.symlink_to(src.resolve())
                n_link += 1
        except OSError as e:
            console.print(f"[red]fail[/red] {src.name}: {e}")
            n_fail += 1

    rel_out = output.relative_to(CFG.root) if output.is_relative_to(CFG.root) else output
    console.print()
    console.print("[green]done[/green]")
    console.print(f"  total chosen:    {len(chosen)}")
    console.print(f"  golden pinned:   {len(golden_paths)}")
    console.print(f"  random fill:     {len(chosen) - len(golden_paths)}")
    if not args.copy:
        console.print(f"  symlinks made:   {n_link}")
    else:
        console.print(f"  files copied:    {n_copy}")
    console.print(f"  already present: {n_skip}")
    if n_fail:
        console.print(f"  failed:          {n_fail}")
    console.print(f"  output dir:      {rel_out}")
    console.print()
    console.print("[cyan]next[/cyan]:")
    console.print(f"  1. set [bold]PDF_DIR={rel_out}[/bold] in .env")
    console.print("  2. in rag>: [bold]/ingest --reset[/bold]   (only the sample gets ingested)")
    console.print("  3. in rag>: [bold]/evaluate[/bold]         (golden docs are guaranteed present)")
    console.print()
    console.print("[dim]to revert: edit PDF_DIR back, run /ingest --reset[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
