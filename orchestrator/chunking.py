"""Built-in chunking utilities for the builtin_chunking feature flag."""

from __future__ import annotations

import re


def chunk_by_section(text: str) -> list[str]:
    """Split text on === or ### headers, returning non-empty sections."""
    parts = re.split(r"\n(?=={3,}|#{3,}\s)", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_paragraph(text: str, min_length: int = 100) -> list[str]:
    """Split on blank lines, merging short paragraphs into the next."""
    raw = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for para in raw:
        para = para.strip()
        if not para:
            continue
        buf = f"{buf}\n\n{para}" if buf else para
        if len(buf) >= min_length:
            chunks.append(buf)
            buf = ""
    if buf:
        if chunks:
            chunks[-1] = f"{chunks[-1]}\n\n{buf}"
        else:
            chunks.append(buf)
    return chunks


def chunk_by_tokens(text: str, chunk_size: int = 6000, overlap: int = 500) -> list[str]:
    """Overlapping character windows (token ~ 4 chars, so default ~1500 tokens)."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
