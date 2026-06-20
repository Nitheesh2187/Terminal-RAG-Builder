"""Answer generation via Groq (OpenAI-compatible API)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from .config import CFG
from .metrics import LatencyRecord, stage
from .retrieve import Hit

SYSTEM_PROMPT = (
    "You are a research assistant answering questions strictly from the provided arXiv excerpts. "
    "Cite chunks inline using [#] markers that match the numbered excerpts. "
    "If the excerpts do not answer the question, say so plainly. Be precise and concise."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not CFG.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return OpenAI(api_key=CFG.groq_api_key, base_url=CFG.groq_base_url)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h.doc_id}" + (f" — {h.title}" if h.title else "")
        blocks.append(f"{head}\n{h.content.strip()}")
    return "\n\n".join(blocks)


@dataclass
class Answer:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


def generate_answer(query: str, hits: list[Hit], *, rec: LatencyRecord) -> Answer:
    context = _format_context(hits)
    user_msg = (
        f"Question: {query}\n\n"
        f"Excerpts:\n{context}\n\n"
        "Answer the question using only the excerpts above. Cite with [#]."
    )
    with stage(rec, "llm_generate"):
        resp = _client().chat.completions.create(
            model=CFG.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
    usage = resp.usage
    return Answer(
        text=resp.choices[0].message.content or "",
        model=resp.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
