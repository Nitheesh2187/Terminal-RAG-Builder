#!/usr/bin/env python3
"""
Extract a custom-sized random subset of arXiv PDFs from the metadata snapshot.

Reads Data/arxiv-metadata-oai-snapshot.json (JSONL), reservoir-samples N
records, downloads their PDFs from arxiv.org, and writes ONE metadata file
recording dataset-wide statistics plus the per-PDF download log.

Usage:
    python extract.py --count 100
    python extract.py --count 500 --seed 7 --output pdfs --delay 3.0
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Data" / "arxiv-metadata-oai-snapshot.json"
STATS_CACHE = ROOT / "Data" / ".stats_cache.json"
USER_AGENT = "Terminal-RAG-Builder/1.0 (mailto:nitheesh.raju@skylarklabs.ai)"


def primary_category(cats: str) -> str:
    return cats.split()[0] if cats else "unknown"


def year_of(rec: dict) -> str:
    versions = rec.get("versions") or []
    if versions:
        try:
            return datetime.strptime(
                versions[0]["created"], "%a, %d %b %Y %H:%M:%S %Z"
            ).strftime("%Y")
        except (KeyError, ValueError):
            pass
    upd = rec.get("update_date") or ""
    return upd[:4] if len(upd) >= 4 else "unknown"


def cache_key() -> dict:
    st = DATASET.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def load_cached_stats():
    if not STATS_CACHE.exists():
        return None
    try:
        cached = json.loads(STATS_CACHE.read_text())
    except json.JSONDecodeError:
        return None
    if cached.get("_key") == cache_key():
        return cached
    return None


def save_cached_stats(stats: dict) -> None:
    payload = {"_key": cache_key(), **stats}
    STATS_CACHE.write_text(json.dumps(payload, indent=2))


def scan_and_sample(count: int, seed: int):
    """One streaming pass: compute full-dataset stats AND reservoir-sample N records.

    If stats are already cached for this exact file, we still re-scan because we
    need to sample — but we skip cache write. (Sampling is the expensive thing
    we cannot cache because the seed changes between runs.)
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []

    total = 0
    primary_counts: Counter = Counter()
    all_cat_counts: Counter = Counter()
    year_counts: Counter = Counter()
    has_doi = 0
    has_journal_ref = 0
    has_license = 0
    multi_version = 0

    start = time.time()
    with DATASET.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            cats = rec.get("categories") or ""
            primary_counts[primary_category(cats)] += 1
            for c in cats.split():
                all_cat_counts[c] += 1
            year_counts[year_of(rec)] += 1
            if rec.get("doi"):
                has_doi += 1
            if rec.get("journal-ref"):
                has_journal_ref += 1
            if rec.get("license"):
                has_license += 1
            if len(rec.get("versions") or []) > 1:
                multi_version += 1

            # Reservoir sampling (algorithm R)
            if len(reservoir) < count:
                reservoir.append(rec)
            else:
                j = rng.randint(0, total - 1)
                if j < count:
                    reservoir[j] = rec

            if total % 250_000 == 0:
                print(
                    f"  scanned {total:,} records "
                    f"({time.time() - start:.1f}s elapsed)",
                    file=sys.stderr,
                )

    stats = {
        "source_file": str(DATASET),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "scan_seconds": round(time.time() - start, 2),
        "total_records": total,
        "unique_primary_categories": len(primary_counts),
        "unique_categories": len(all_cat_counts),
        "records_with_doi": has_doi,
        "records_with_journal_ref": has_journal_ref,
        "records_with_license": has_license,
        "records_with_multiple_versions": multi_version,
        "top_primary_categories": dict(primary_counts.most_common(25)),
        "top_categories": dict(all_cat_counts.most_common(25)),
        "records_per_year": dict(sorted(year_counts.items())),
    }
    return stats, reservoir


def get_stats(count: int, seed: int):
    cached = load_cached_stats()
    if cached:
        print(f"Using cached dataset stats (scanned {cached['total_records']:,} records).")
        # Still need to sample — stream once but skip stats work.
        rng = random.Random(seed)
        reservoir: list[dict] = []
        total = 0
        start = time.time()
        with DATASET.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if len(reservoir) < count:
                    reservoir.append(rec)
                else:
                    j = rng.randint(0, total - 1)
                    if j < count:
                        reservoir[j] = rec
                if total % 500_000 == 0:
                    print(
                        f"  sampling pass: {total:,} records "
                        f"({time.time() - start:.1f}s)",
                        file=sys.stderr,
                    )
        return {k: v for k, v in cached.items() if k != "_key"}, reservoir

    print("No cached stats — performing full scan (this takes a few minutes)...")
    stats, reservoir = scan_and_sample(count, seed)
    save_cached_stats(stats)
    return stats, reservoir


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_one(arxiv_id: str, dest: Path, timeout: int = 60) -> dict:
    url = pdf_url(arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return {
            "id": arxiv_id,
            "url": url,
            "path": str(dest.relative_to(ROOT)),
            "bytes": len(data),
            "status": "ok",
            "seconds": round(time.time() - started, 2),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return {
            "id": arxiv_id,
            "url": url,
            "status": "error",
            "error": str(e),
            "seconds": round(time.time() - started, 2),
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", "-n", type=int, required=True,
                   help="Number of PDFs to sample and download.")
    p.add_argument("--output", "-o", default="pdfs",
                   help="Directory to download PDFs into (default: ./pdfs).")
    p.add_argument("--metadata", "-m", default="metadata.json",
                   help="Path to write the combined metadata/stats file.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling (default: 42).")
    p.add_argument("--delay", type=float, default=3.0,
                   help="Seconds between PDF requests (arxiv asks for ~3s; default: 3.0).")
    p.add_argument("--rescan", action="store_true",
                   help="Force re-scan of dataset, ignore stats cache.")
    args = p.parse_args()

    if not DATASET.exists():
        print(f"ERROR: dataset not found at {DATASET}", file=sys.stderr)
        return 1
    if args.count <= 0:
        print("ERROR: --count must be positive", file=sys.stderr)
        return 1

    if args.rescan and STATS_CACHE.exists():
        STATS_CACHE.unlink()

    out_dir = (ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = (ROOT / args.metadata).resolve()

    print(f"Sampling {args.count} record(s) with seed={args.seed}...")
    stats, sample = get_stats(args.count, args.seed)
    print(f"Dataset has {stats['total_records']:,} records; sampled {len(sample)}.")

    print(f"Downloading PDFs to {out_dir} (delay={args.delay}s between requests)...")
    downloads = []
    for i, rec in enumerate(sample, 1):
        aid = rec["id"]
        safe = aid.replace("/", "_")
        dest = out_dir / f"{safe}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            downloads.append({
                "id": aid,
                "url": pdf_url(aid),
                "path": str(dest.relative_to(ROOT)),
                "bytes": dest.stat().st_size,
                "status": "cached",
                "seconds": 0.0,
            })
            print(f"  [{i}/{len(sample)}] {aid} — already on disk, skipping")
            continue
        result = download_one(aid, dest)
        downloads.append(result)
        status_marker = "OK" if result["status"] == "ok" else "FAIL"
        print(f"  [{i}/{len(sample)}] {aid} — {status_marker} ({result['seconds']}s)")
        if i < len(sample):
            time.sleep(args.delay)

    successes = [d for d in downloads if d["status"] in ("ok", "cached")]
    failures = [d for d in downloads if d["status"] == "error"]

    # Build per-sample metadata (record fields + download outcome)
    by_id = {d["id"]: d for d in downloads}
    sample_meta = []
    for rec in sample:
        sample_meta.append({
            "id": rec["id"],
            "title": (rec.get("title") or "").strip(),
            "authors": rec.get("authors"),
            "categories": rec.get("categories"),
            "primary_category": primary_category(rec.get("categories") or ""),
            "year": year_of(rec),
            "doi": rec.get("doi"),
            "abstract": (rec.get("abstract") or "").strip(),
            "download": by_id.get(rec["id"]),
        })

    subset_category_counts = Counter(s["primary_category"] for s in sample_meta)
    subset_year_counts = Counter(s["year"] for s in sample_meta)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "extraction_params": {
            "count_requested": args.count,
            "seed": args.seed,
            "delay_seconds": args.delay,
            "output_dir": str(out_dir.relative_to(ROOT)),
        },
        "dataset_statistics": stats,
        "extraction_log": {
            "requested": len(sample),
            "succeeded": len(successes),
            "failed": len(failures),
            "subset_primary_category_counts": dict(subset_category_counts.most_common()),
            "subset_year_counts": dict(sorted(subset_year_counts.items())),
            "failures": [{"id": f["id"], "error": f["error"]} for f in failures],
        },
        "samples": sample_meta,
    }

    meta_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nMetadata written to {meta_path}")
    print(f"Success: {len(successes)} / {len(sample)} "
          f"(failed: {len(failures)})")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
