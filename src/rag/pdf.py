"""PDF text + section extraction.

Two entry points:
  - pdf_to_text(path)      → str           (whole-document text, flat)
  - pdf_to_sections(path)  → list[Section] (title + body per section)

Section extraction tries the PDF's embedded TOC first (most reliable, when
present), then falls back to regex heuristics over the flat text. If nothing
looks like a section, the whole document comes back as one Section(None, text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Section:
    title: str | None
    text: str


def pdf_to_text(path: Path) -> str:
    """Whole-document text. Strips NUL bytes for Postgres-safety."""
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
    return text.replace("\x00", "")


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

# Common header anchors in academic papers — case-insensitive.
KNOWN_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "preliminaries", "problem statement", "motivation",
    "methodology", "methods", "method", "approach", "model",
    "framework", "architecture", "implementation",
    "experiments", "experimental setup", "experimental results",
    "evaluation", "results", "analysis", "ablation studies", "ablations",
    "discussion", "limitations", "future work",
    "conclusion", "conclusions",
    "references", "appendix", "acknowledgments", "acknowledgements",
]

# Patterns for header detection in flat text.
# Both REQUIRE a blank line before the header — real section titles are set
# off by whitespace; mid-paragraph mentions, table/figure captions, and
# inline references aren't, which is how they sneak past looser regexes.
#
# 1. Numbered headers: "1 Introduction", "2.1 Related Work", "3.1.4 Subsection"
RE_NUMBERED = re.compile(
    r"(?:\n\n|\A)[ \t]*(\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+"
    r"([A-Z][A-Za-z][A-Za-z\- ]{2,70}?)\s*\n",
)
# 2. Known anchors: "Abstract", "Introduction", "Conclusion"
RE_KNOWN = re.compile(
    r"(?:\n\n|\A)[ \t]*(" + "|".join(re.escape(s) for s in KNOWN_SECTIONS) + r")\s*\n",
    re.IGNORECASE,
)

# Reject titles that look like caption / reference fragments.
_BAD_TITLE_TOKENS = re.compile(
    r"\b(TABLE|FIG|FIGURE|EQ|EQUATION|REF|PROOF|LEMMA|THEOREM|COROLLARY|"
    r"PROPOSITION|REMARK|DEFINITION)\b",
    re.IGNORECASE,
)


def _looks_like_section_title(title: str) -> bool:
    t = title.strip()
    if not (3 <= len(t) <= 80):
        return False
    # All-caps short tokens are usually inline labels (TABLE I, FIG. 3), not titles
    letters = [c for c in t if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return False
    # Captions / proofs / etc.
    if _BAD_TITLE_TOKENS.search(t):
        return False
    # Digits past the first few chars usually mean equation / page refs
    if any(c.isdigit() for c in t[3:]):
        return False
    return True


def _normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _split_by_toc(doc: fitz.Document) -> list[Section]:
    """Use the PDF's embedded outline to split text into sections.

    Approach: for each TOC entry, locate the title string inside the full text
    (searching forward from its page). Sections span title-to-title.
    """
    toc = doc.get_toc()
    if not toc:
        return []

    page_texts = [page.get_text("text") for page in doc]
    full = "\n".join(page_texts)
    # Page boundaries in the joined full text
    page_starts = [0]
    for t in page_texts[:-1]:
        page_starts.append(page_starts[-1] + len(t) + 1)  # +1 for the "\n"

    positions: list[tuple[int, str]] = []
    cursor = 0
    for _level, raw_title, page in toc:
        title = _normalize_title(raw_title)
        # Filter out clearly-broken TOC entries (some PDFs have garbage bookmarks)
        if not title or len(title) < 3 or len(title) > 100:
            continue
        if not _looks_like_section_title(title):
            continue
        pidx = max(0, min(page - 1, len(page_starts) - 1))
        search_from = max(cursor, page_starts[pidx] - 50)
        idx = _find_title(full, title, search_from)
        if idx >= cursor:
            positions.append((idx, title))
            cursor = idx + len(title)

    if len(positions) < 2:
        return []

    sections: list[Section] = []
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full)
        body_start = start + len(title)
        body = full[body_start:end].strip()
        if body:
            sections.append(Section(title=title, text=body))
    return sections


def _find_title(haystack: str, title: str, start: int) -> int:
    """Locate title in haystack, tolerant of whitespace variation."""
    # Direct find first (cheap path)
    idx = haystack.find(title, start)
    if idx >= 0:
        return idx
    # Loose: collapse whitespace in both, find, then map back roughly
    title_loose = re.sub(r"\s+", " ", title)
    # Walk a sliding window of the haystack with whitespace collapsed —
    # this is approximate. Fall back to first non-whitespace prefix match.
    prefix = title_loose[:30]
    return haystack.find(prefix, start) if prefix else -1


def _split_by_regex(text: str) -> list[Section]:
    """Heuristic section split for PDFs without a usable TOC."""
    cands: list[tuple[int, int, str]] = []  # (start, end_of_title_line, title)
    for m in RE_NUMBERED.finditer(text):
        title_body = m.group(2).strip()
        if not _looks_like_section_title(title_body):
            continue
        title = _normalize_title(f"{m.group(1)} {title_body}")
        cands.append((m.start(), m.end(), title))
    for m in RE_KNOWN.finditer(text):
        title = _normalize_title(m.group(1)).title()
        cands.append((m.start(), m.end(), title))

    if not cands:
        return []

    cands.sort()
    deduped: list[tuple[int, int, str]] = []
    for c in cands:
        if deduped and c[0] - deduped[-1][0] < 5:
            continue
        deduped.append(c)
    if len(deduped) < 2:
        return []

    sections: list[Section] = []
    for i, (_, body_start, title) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        body = text[body_start:end].strip()
        # Drop sections with implausibly short body — likely caption fragments.
        if len(body) < 200:
            continue
        sections.append(Section(title=title, text=body))
    return sections


def pdf_to_sections(path: Path) -> list[Section]:
    """Return sections of a PDF.

    Tries embedded TOC → regex heuristic → single-section fallback (the whole
    text as one section with title=None). Always returns ≥1 section so callers
    don't have to handle empty.
    """
    with fitz.open(path) as doc:
        # Try TOC
        toc_sections = _split_by_toc(doc)
        if len(toc_sections) >= 2:
            for s in toc_sections:
                s.text = s.text.replace("\x00", "")
            return toc_sections
        # Fall back to flat text + regex
        flat = "\n".join(page.get_text("text") for page in doc).replace("\x00", "")
    rx_sections = _split_by_regex(flat)
    if len(rx_sections) >= 2:
        return rx_sections
    # Last resort: whole doc as one section
    return [Section(title=None, text=flat)]
