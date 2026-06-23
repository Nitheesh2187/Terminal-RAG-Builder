"""Token-aware recursive text chunking."""
from __future__ import annotations

import tiktoken

from .models import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _split_recursive(text: str, max_toks: int, seps: list[str]) -> list[str]:
    if not text.strip():
        return []
    if len(_ENCODER.encode(text)) <= max_toks:
        return [text]
    if not seps:
        ids = _ENCODER.encode(text)
        return [
            _ENCODER.decode(ids[i : i + max_toks])
            for i in range(0, len(ids), max_toks)
        ]
    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep)
    out: list[str] = []
    buf = ""
    for p in pieces:
        candidate = buf + (sep if buf else "") + p
        if len(_ENCODER.encode(candidate)) <= max_toks:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if len(_ENCODER.encode(p)) > max_toks:
                out.extend(_split_recursive(p, max_toks, rest))
                buf = ""
            else:
                buf = p
    if buf:
        out.append(buf)
    return out


def chunk_sections(sections, *, max_tokens: int = 800, overlap: int = 100) -> list[Chunk]:
    """Chunk each section independently, preserving the section title on every chunk.

    Long sections get sub-chunked with the same recursive token-aware splitter.
    Chunk indices are reassigned globally across the document so they stay unique.
    """
    out: list[Chunk] = []
    for s in sections:
        title = getattr(s, "title", None) if not isinstance(s, tuple) else s[0]
        text = getattr(s, "text", None) if not isinstance(s, tuple) else s[1]
        if not text:
            continue
        sub = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
        for c in sub:
            c.section = title
            out.append(c)
    # Re-number globally so chunk_idx is unique within the doc
    for new_idx, c in enumerate(out):
        c.idx = new_idx
    return out


def chunk_text(text: str, *, max_tokens: int = 800, overlap: int = 100) -> list[Chunk]:
    """Recursive split, then add token overlap between adjacent chunks."""
    base = _split_recursive(text, max_tokens, SEPARATORS)
    if not base:
        return []
    if overlap <= 0 or len(base) == 1:
        return [Chunk(i, t, len(_ENCODER.encode(t))) for i, t in enumerate(base)]

    out: list[Chunk] = []
    prev_tail_ids: list[int] = []
    for i, t in enumerate(base):
        if prev_tail_ids:
            merged_ids = prev_tail_ids + _ENCODER.encode(t)
            t_out = _ENCODER.decode(merged_ids)
        else:
            t_out = t
        ids = _ENCODER.encode(t_out)
        out.append(Chunk(i, t_out, len(ids)))
        tail_ids = _ENCODER.encode(t)[-overlap:]
        prev_tail_ids = tail_ids
    return out
