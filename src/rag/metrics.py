"""Latency tracking + retrieval metrics + Ragas metric computation + Rich rendering."""
from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from functools import lru_cache

from rich.console import Console
from rich.table import Table

from .config import CFG
from .models import LatencyRecord

console = Console()


# ---------------------------------------------------------------------------
# Classical retrieval metrics (doc-id based, no LLM, deterministic)
# ---------------------------------------------------------------------------

def retrieval_metrics(retrieved_doc_ids: list[str], gold: list[str], k: int) -> dict[str, float]:
    """Hit@1, Recall@k, MRR@k, nDCG@k. Gold treated as the binary-relevant set.

    Caller is responsible for deduping `retrieved_doc_ids` if doc-level
    metrics are intended — duplicates inflate nDCG above 1.0.
    """
    gold_set = set(gold)
    topk = retrieved_doc_ids[:k]
    if not gold_set:
        return {"hit@1": 0.0, "recall@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0}

    hit_at_1 = 1.0 if topk and topk[0] in gold_set else 0.0
    recall = len(gold_set.intersection(topk)) / len(gold_set)

    mrr = 0.0
    for i, d in enumerate(topk, 1):
        if d in gold_set:
            mrr = 1.0 / i
            break

    dcg = sum((1.0 / math.log2(i + 1)) for i, d in enumerate(topk, 1) if d in gold_set)
    ideal = sum((1.0 / math.log2(i + 1)) for i in range(1, min(len(gold_set), k) + 1))
    ndcg = dcg / ideal if ideal > 0 else 0.0
    return {"hit@1": hit_at_1, "recall@k": recall, "mrr@k": mrr, "ndcg@k": ndcg}


# ---------------------------------------------------------------------------
# No-LLM context precision/recall via n-gram overlap (deterministic, free)
#
# Replaces Ragas' NonLLM* metrics, which compare strings with Levenshtein
# edit-distance and a 0.5 threshold. That collapses toward 0 whenever a
# retrieved chunk and a gold span differ in length — even when the chunk
# fully contains the span — so it measured byte-equality, not retrieval.
#
# Here we compare *n-gram sets*, which is robust to chunk boundaries and to
# PDF-extraction artifacts (hyphenation, ligatures), and is symmetric in
# length. No LLM, no API, no rate limits.
# ---------------------------------------------------------------------------

def _normalize_tokens(text: str) -> list[str]:
    """Lowercase, rejoin PDF hyphenation ('architec- tural' -> 'architectural'),
    then split into alphanumeric word tokens. Tolerant enough that text from two
    different PDF parsers still overlaps."""
    if not text:
        return []
    t = text.lower().replace("- ", "")   # join hyphenated line/word breaks
    tokens: list[str] = []
    cur: list[str] = []
    for ch in t:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def _ngrams(tokens: list[str], n: int) -> set[tuple]:
    if not tokens:
        return set()
    if len(tokens) <= n:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _coverage(a: set, b: set) -> float:
    """Fraction of `a` found in `b`. 0 if `a` is empty."""
    return len(a & b) / len(a) if a else 0.0


def _overlap_coef(a: set, b: set) -> float:
    """Szymkiewicz-Simpson overlap coefficient: |a∩b| / min(|a|,|b|).
    Symmetric in length — high when the smaller set is largely contained in
    the larger, so a small gold span inside a big chunk (and vice versa) both
    score high. This is the relevance test for context precision."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def context_overlap_metrics(
    samples: list[dict],
    *,
    n: int | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """No-LLM context precision/recall via n-gram overlap.

    Each sample: {question, contexts: [chunk_text...], reference_contexts: [span...]}

    Definitions (per query, then averaged):
      recall    — fraction of gold spans 'covered': a span is covered if
                  |span_ngrams ∩ (union of retrieved chunk ngrams)| / |span_ngrams|
                  >= threshold. (Union, so a span split across adjacent chunks
                  still counts.)
      precision — fraction of retrieved chunks that are 'relevant': a chunk is
                  relevant if its overlap coefficient with some gold span
                  >= threshold.

    Also returns *_soft (the same scores without thresholding, mean of the
    continuous values) — use these to compare chunking strategies without a
    threshold artifact, and to calibrate `threshold`.
    """
    n = n or CFG.ctx_overlap_n
    threshold = threshold if threshold is not None else CFG.ctx_overlap_threshold

    prec_bin, rec_bin, prec_soft, rec_soft = [], [], [], []
    mrr_list, map_list = [], []
    n_scored = 0
    for s in samples:
        spans = [_ngrams(_normalize_tokens(r), n) for r in (s.get("reference_contexts") or [])]
        spans = [sp for sp in spans if sp]
        if not spans:
            continue
        n_scored += 1
        chunks = [_ngrams(_normalize_tokens(c), n) for c in (s.get("contexts") or [])]

        union: set = set().union(*chunks) if chunks else set()
        span_cov = [_coverage(sp, union) for sp in spans]
        rec_soft.append(sum(span_cov) / len(span_cov))
        rec_bin.append(sum(1 for c in span_cov if c >= threshold) / len(span_cov))

        if chunks:
            # chunks arrive in rank order (hybrid_search ranking preserved).
            chunk_rel = [max((_overlap_coef(ch, sp) for sp in spans), default=0.0) for ch in chunks]
            prec_soft.append(sum(chunk_rel) / len(chunk_rel))
            prec_bin.append(sum(1 for c in chunk_rel if c >= threshold) / len(chunk_rel))

            # Rank-aware, sparse-label-robust precision proxies. Unlike
            # precision@k these do NOT degrade when you raise top_k: an
            # irrelevant chunk added at the bottom changes neither.
            rel_flags = [1 if c >= threshold else 0 for c in chunk_rel]
            first = next((i for i, r in enumerate(rel_flags, 1) if r), 0)
            mrr_list.append(1.0 / first if first else 0.0)
            n_rel = sum(rel_flags)
            if n_rel:
                running, ap = 0, 0.0
                for i, r in enumerate(rel_flags, 1):
                    if r:
                        running += 1
                        ap += running / i
                map_list.append(ap / n_rel)
            else:
                map_list.append(0.0)
        else:
            prec_soft.append(0.0)
            prec_bin.append(0.0)
            mrr_list.append(0.0)
            map_list.append(0.0)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "context_precision": _mean(prec_bin),
        "context_recall": _mean(rec_bin),
        "context_precision_soft": _mean(prec_soft),
        "context_recall_soft": _mean(rec_soft),
        "chunk_mrr": _mean(mrr_list),
        "chunk_map": _mean(map_list),
        "n_scored": float(n_scored),
        "threshold": float(threshold),
        "ngram_n": float(n),
    }


def context_overlap_detail(sample: dict, *, n: int | None = None) -> list[float]:
    """Per-span union-coverage for one sample — for threshold calibration."""
    n = n or CFG.ctx_overlap_n
    spans = [_ngrams(_normalize_tokens(r), n) for r in (sample.get("reference_contexts") or [])]
    spans = [sp for sp in spans if sp]
    chunks = [_ngrams(_normalize_tokens(c), n) for c in (sample.get("contexts") or [])]
    union: set = set().union(*chunks) if chunks else set()
    return [_coverage(sp, union) for sp in spans]


def render_context_overlap(scores: dict[str, float]) -> None:
    if not scores:
        console.print("[dim]no context-overlap scores to render[/dim]")
        return
    table = Table(
        title=(
            f"context metrics (no-LLM n-gram overlap, "
            f"n={int(scores.get('ngram_n', 0))}, thr={scores.get('threshold', 0):.2f}, "
            f"queries={int(scores.get('n_scored', 0))})"
        ),
        show_lines=False,
    )
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right", style="green")
    for key in ("context_precision", "context_recall",
                "context_precision_soft", "context_recall_soft",
                "chunk_mrr", "chunk_map"):
        if key in scores:
            table.add_row(key, f"{scores[key]:.4f}")
    console.print(table)


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


def compute_ragas_metrics(
    samples: list[dict],
    *,
    include_llm_metrics: bool = False,
) -> dict[str, float]:
    """Compute Ragas metrics on per-query samples.

    Each sample:
        {question, contexts, ground_truth?, reference_contexts?, response?}

    What runs:
      Non-LLM (always when applicable, free, deterministic):
        - non_llm_context_precision_with_reference  (needs reference_contexts)
        - non_llm_context_recall                    (needs reference_contexts)

      LLM-judged (only when include_llm_metrics=True — burns LLM calls):
        - faithfulness          (needs response)
        - answer_relevancy      (needs response)
        - context_precision     (needs ground_truth)
        - context_recall        (needs ground_truth)
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

    any_response = any(s.get("response") for s in samples)
    any_gt = any(s.get("ground_truth") for s in samples)
    any_ref_ctx = any(s.get("reference_contexts") for s in samples)

    metrics = []

    # Non-LLM: string-similarity, no rate-limit pressure.
    if any_ref_ctx:
        try:
            from ragas.metrics import (
                NonLLMContextPrecisionWithReference,
                NonLLMContextRecall,
            )
            metrics += [
                NonLLMContextPrecisionWithReference(),
                NonLLMContextRecall(),
            ]
        except ImportError:
            console.print(
                "[yellow]warn[/yellow]: non-LLM context metrics unavailable "
                "in this ragas version — install ragas>=0.2.10"
            )

    # LLM-judged — only when explicitly opted into.
    if include_llm_metrics:
        if any_response:
            metrics += [faithfulness, answer_relevancy]
        if any_gt:
            metrics += [context_precision, context_recall]

    if not metrics:
        console.print("[dim]ragas: no applicable metrics for this sample shape[/dim]")
        return {}

    rows = []
    for s in samples:
        row = {
            "user_input": s["question"],
            "retrieved_contexts": list(s.get("contexts") or []),
        }
        if any_response:
            row["response"] = s.get("response") or ""
        if any_gt:
            row["reference"] = s.get("ground_truth") or ""
        if any_ref_ctx:
            row["reference_contexts"] = list(s.get("reference_contexts") or [])
        rows.append(row)

    needs_llm = include_llm_metrics and any(
        m in metrics for m in [faithfulness, answer_relevancy, context_precision, context_recall]
    )

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=build_ragas_llm() if needs_llm else None,
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
