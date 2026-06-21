"""Run the RAG pipeline on a golden set and print retrieval + latency metrics."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import CFG
from .generate import generate_answer
from .metrics import compute_ragas_metrics, percentiles, render_latency, render_ragas_metrics
from .models import GoldenItem, LatencyRecord
from .retrieve import hybrid_search

console = Console()


def load_golden(path: Path | str) -> list[GoldenItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden file not found: {p}")
    items: list[GoldenItem] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items.append(GoldenItem(
                question=r["question"],
                gold_doc_ids=list(r.get("gold_doc_ids") or []),
                gold_answer=r.get("gold_answer"),
            ))
    return items


def _retrieval_metrics(retrieved_doc_ids: list[str], gold: list[str], k: int) -> dict[str, float]:
    """Hit@1, Recall@k, MRR@k, nDCG@k. Gold treated as relevant set."""
    gold_set = set(gold)
    topk = retrieved_doc_ids[:k]
    if not gold_set:
        return {"hit@1": 0.0, "recall@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0}

    hit_at_1 = 1.0 if topk and topk[0] in gold_set else 0.0
    recalled = len(gold_set.intersection(topk))
    recall = recalled / len(gold_set)

    mrr = 0.0
    for i, d in enumerate(topk, 1):
        if d in gold_set:
            mrr = 1.0 / i
            break

    dcg = sum((1.0 / math.log2(i + 1)) for i, d in enumerate(topk, 1) if d in gold_set)
    ideal = sum((1.0 / math.log2(i + 1)) for i in range(1, min(len(gold_set), k) + 1))
    ndcg = dcg / ideal if ideal > 0 else 0.0
    return {"hit@1": hit_at_1, "recall@k": recall, "mrr@k": mrr, "ndcg@k": ndcg}


def run_evaluation(*, golden_path: Path | str | None = None,
                   k: int | None = None,
                   with_generation: bool = False,
                   with_ragas: bool = False) -> dict:
    golden_path = Path(golden_path or CFG.golden_path)
    k = k or CFG.top_k
    items = load_golden(golden_path)
    if not items:
        console.print("[yellow]golden set is empty[/yellow]")
        return {}

    # Ragas needs generated answers
    if with_ragas and not with_generation:
        console.print("[dim]--ragas implies generation; enabling[/dim]")
        with_generation = True

    per_query_latencies: dict[str, list[float]] = {
        "embed_query": [], "dense_sql": [], "sparse_sql": [], "rrf_fuse": [], "llm_generate": [], "total": [],
    }
    metrics_acc = {"hit@1": [], "recall@k": [], "mrr@k": [], "ndcg@k": []}
    ragas_samples: list[dict] = []

    console.print(
        f"[cyan]evaluate[/cyan]: {len(items)} queries, k={k}, "
        f"generate={with_generation}, ragas={with_ragas}"
    )
    t_overall = time.perf_counter()
    for item in items:
        rec = LatencyRecord(command="/evaluate.query")
        hits = hybrid_search(item.question, k=k, rec=rec)
        retrieved_docs = [h.doc_id for h in hits]
        m = _retrieval_metrics(retrieved_docs, item.gold_doc_ids, k)
        for key, val in m.items():
            metrics_acc[key].append(val)

        ans_text: str | None = None
        if with_generation:
            try:
                ans = generate_answer(item.question, hits, rec=rec)
                ans_text = ans.text
            except Exception as e:
                console.print(f"[red]gen fail[/red]: {e}")

        if with_ragas and ans_text is not None:
            ragas_samples.append({
                "question": item.question,
                "answer": ans_text,
                "contexts": [h.content for h in hits],
                "ground_truth": item.gold_answer,
            })

        for s in rec.stages:
            per_query_latencies.setdefault(s.name, []).append(s.ms)
        per_query_latencies["total"].append(rec.total_ms)

    wall_s = time.perf_counter() - t_overall

    metrics_table = Table(title="retrieval metrics", show_lines=False)
    metrics_table.add_column("metric", style="cyan")
    metrics_table.add_column("mean", justify="right", style="green")
    metrics_table.add_column("n", justify="right", style="dim")
    for key, vals in metrics_acc.items():
        mean = sum(vals) / len(vals) if vals else 0.0
        metrics_table.add_row(key, f"{mean:.4f}", str(len(vals)))
    console.print(metrics_table)

    lat_table = Table(title="latency (ms) — per query", show_lines=False)
    lat_table.add_column("stage", style="cyan")
    lat_table.add_column("p50", justify="right", style="green")
    lat_table.add_column("p95", justify="right", style="yellow")
    lat_table.add_column("p99", justify="right", style="red")
    lat_table.add_column("mean", justify="right", style="dim")
    lat_table.add_column("max", justify="right", style="dim")
    for stage_name, vals in per_query_latencies.items():
        if not vals:
            continue
        p = percentiles(vals)
        lat_table.add_row(
            stage_name,
            f"{p['p50']:.1f}", f"{p['p95']:.1f}", f"{p['p99']:.1f}",
            f"{p['mean']:.1f}", f"{p['max']:.1f}",
        )
    console.print(lat_table)
    console.print(f"[dim]wall time: {wall_s:.2f}s[/dim]")

    summary_rec = LatencyRecord(command="/evaluate (totals)")
    summary_rec.set("queries", len(items))
    summary_rec.set("wall_seconds", wall_s)
    summary_rec.set("queries_per_sec", len(items) / wall_s if wall_s > 0 else 0.0)
    render_latency(summary_rec, title="/evaluate — summary")

    ragas_scores: dict[str, float] = {}
    if with_ragas:
        if not ragas_samples:
            console.print("[yellow]ragas: no samples (generation may have failed)[/yellow]")
        else:
            console.print(f"[cyan]ragas[/cyan]: scoring {len(ragas_samples)} samples (LLM calls — slow)")
            t_r = time.perf_counter()
            try:
                ragas_scores = compute_ragas_metrics(ragas_samples)
                console.print(f"[dim]ragas wall: {time.perf_counter() - t_r:.1f}s[/dim]")
                render_ragas_metrics(ragas_scores)
            except Exception as e:
                console.print(f"[red]ragas failed[/red]: {e}")

    return {
        "metrics": {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics_acc.items()},
        "latencies_ms": {k: percentiles(v) for k, v in per_query_latencies.items() if v},
        "ragas": ragas_scores,
        "queries": len(items),
        "wall_seconds": wall_s,
    }
