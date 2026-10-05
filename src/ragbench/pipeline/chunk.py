"""Split ingested documents into chunks for vector and hybrid search.

Every chunk records exactly where it came from: character offsets into the document
text (so `text[start:end] == chunk.text` always holds) and the page(s) it spans. That
lets the benchmark check whether a retrieved chunk is on the right page, and lets
anyone audit what a search method actually saw.

The agent searches the full markdown files instead, so chunking only affects the
vector and hybrid methods. Chunk size is therefore a parameter of the comparison, and
every result records the settings it was run with.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

DEFAULT_SIZE = 1000  # characters, roughly 200-250 English words
DEFAULT_OVERLAP = 200  # characters shared between neighbouring chunks


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str  # "<doc_id>:<index>", e.g. "bdfaa68d8984:0007"
    start: int  # character offset in the document text, inclusive
    end: int  # character offset, exclusive
    page: int  # 1-based page where the chunk starts
    page_end: int  # 1-based page where the chunk ends
    text: str


def page_at(offset: int, page_offsets: list[int]) -> int:
    """1-based page number containing character `offset`.

    `page_offsets[i]` is where page i+1 starts, so the page is the number of page
    starts at or before the offset.
    """
    return max(1, bisect_right(page_offsets, offset))


def chunk_fixed(
    doc_id: str,
    text: str,
    page_offsets: list[int],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Cut `text` into windows of `size` characters, each overlapping the previous one.

    The windows ignore words and sentences on purpose: this is the simplest baseline.
    The final chunk may be shorter than `size`, but it is never a pure repeat of the
    previous chunk's overlap.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap}, size={size}")

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:{len(chunks):04d}",
                start=start,
                end=end,
                page=page_at(start, page_offsets),
                page_end=page_at(end - 1, page_offsets),
                text=text[start:end],
            )
        )
        if end == len(text):
            break
        start += step
    return chunks
