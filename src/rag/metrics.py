"""Latency tracking + Rich rendering for every command."""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class StageTiming:
    name: str
    ms: float


@dataclass
class LatencyRecord:
    command: str
    stages: list[StageTiming] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, ms: float) -> None:
        self.stages.append(StageTiming(name, ms))

    def set(self, key: str, value: float) -> None:
        self.counters[key] = value

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)


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
