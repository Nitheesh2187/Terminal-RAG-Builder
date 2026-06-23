"""Project-wide dataclasses (no behavior beyond simple accessors)."""
from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class Chunk:
    idx: int
    text: str
    n_tokens: int
    section: str | None = None


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    chunk_idx: int
    content: str
    title: str | None
    section: str | None
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float
    rerank_score: float | None = None


@dataclass
class Answer:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class GoldenItem:
    question: str
    gold_doc_ids: list[str]
    gold_answer: str | None = None
    reference_contexts: list[str] = field(default_factory=list)
