"""Latency tracking + Ragas metric computation + Rich rendering."""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from functools import lru_cache

from rich.console import Console
from rich.table import Table

from .config import CFG
from .models import LatencyRecord

console = Console()


@contextmanager
def stage(rec: LatencyRecord, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec.add(name, (time.perf_counter() - t0) * 1000.0)


def render_latency(rec: LatencyRecord, *, title: str | None = None) -> None:
    title = title or f"{rec.command} — latency"
    table = Table(title=title, show_lines=False, expand=False)
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Time (ms)", justify="right", style="green")
    table.add_column("% of total", justify="right", style="dim")
    total = rec.total_ms or 1.0
    for s in rec.stages:
        table.add_row(s.name, f"{s.ms:,.2f}", f"{(s.ms / total) * 100:5.1f}%")
    table.add_section()
    table.add_row("TOTAL", f"{rec.total_ms:,.2f}", "100.0%")
    console.print(table)

    if rec.counters:
        ctable = Table(title="counters", show_lines=False, expand=False)
        ctable.add_column("metric", style="cyan")
        ctable.add_column("value", justify="right", style="green")
        for k, v in rec.counters.items():
            if isinstance(v, float):
                ctable.add_row(k, f"{v:,.3f}")
            else:
                ctable.add_row(k, f"{v:,}")
        console.print(ctable)


# ---------------------------------------------------------------------------
# Ragas integration: shared LangChain wrappers + metric computation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_ragas_llm():
    """Wrap Groq as a Ragas-compatible LangChain LLM (deferred import)."""
    if not CFG.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper
    llm = ChatOpenAI(
        model=CFG.groq_model,
        api_key=CFG.groq_api_key,
        base_url=CFG.groq_base_url,
        temperature=0.0,
        timeout=60,
        max_retries=3,
    )
    return LangchainLLMWrapper(llm)


@lru_cache(maxsize=1)
def build_ragas_embeddings():
    """Wrap our local bge model as a Ragas-compatible LangChain embedding."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    hf = HuggingFaceEmbeddings(
        model_name=CFG.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


def compute_ragas_metrics(samples: list[dict]) -> dict[str, float]:
    """Compute Ragas metrics on a list of per-query samples.

    Each sample is:
        {question, answer, contexts: list[str], ground_truth: str | None}

    Metrics auto-selected by availability:
      - faithfulness, answer_relevancy:   always (need q, a, contexts)
      - context_precision, context_recall: only when any ground_truth is present
    """
    if not samples:
        return {}
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    any_gt = any(s.get("ground_truth") for s in samples)
    metrics = [faithfulness, answer_relevancy]
    if any_gt:
        metrics += [context_precision, context_recall]

    rows = []
    for s in samples:
        row = {
            "user_input": s["question"],
            "response": s["answer"],
            "retrieved_contexts": list(s.get("contexts") or []),
        }
        gt = s.get("ground_truth")
        if any_gt:
            row["reference"] = gt or ""
        rows.append(row)

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=build_ragas_llm(),
        embeddings=build_ragas_embeddings(),
        show_progress=True,
    )
    df = result.to_pandas()
    out: dict[str, float] = {}
    for m in metrics:
        col = m.name
        if col in df.columns:
            series = df[col].dropna()
            if len(series):
                out[col] = float(series.mean())
    return out


def render_ragas_metrics(scores: dict[str, float]) -> None:
    if not scores:
        console.print("[dim]no ragas scores to render[/dim]")
        return
    table = Table(title="Ragas metrics (mean across queries)", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("mean", justify="right", style="green")
    for k, v in scores.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    s = sorted(values)
    def pct(p):
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)
    return {
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "mean": statistics.fmean(s),
        "max": s[-1],
    }
