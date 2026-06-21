"""PDF text extraction."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_text(path: Path) -> str:
    """Extract concatenated text from every page of a PDF.

    Strips NUL bytes (\\x00) — PyMuPDF can emit them for PDFs with unusual
    font encodings, and Postgres TEXT columns reject them.
    """
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
    return text.replace("\x00", "")
