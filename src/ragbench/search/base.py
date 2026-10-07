"""The contract every search method follows, so the benchmark can treat them identically.

A result says *where* in the corpus the evidence is (document, character range, pages),
not only which chunk it came from. Chunk-based methods (keyword, BM25, vector, hybrid)
fill in `chunk_id`; the agent reads files directly and reports the ranges it read. Both
can then be scored with the same metrics: right document? right page?
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ragbench.pipeline.chunk import Chunk


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    start: int  # character range in the document text: text[start:end]
    end: int
    page: int  # 1-based, where the range starts
    page_end: int  # 1-based, where the range ends
    score: float  # higher is better; only comparable within one method and one query
    text: str
    chunk_id: str = ""  # empty for methods that don't work on chunks (the agent)

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> SearchResult:
        return cls(
            doc_id=chunk.doc_id,
            start=chunk.start,
            end=chunk.end,
            page=chunk.page,
            page_end=chunk.page_end,
            score=score,
            text=chunk.text,
            chunk_id=chunk.chunk_id,
        )


@runtime_checkable
class Searcher(Protocol):
    """Anything with a `name` and a `search` method is a search backend.

    The contract (checked for every backend in tests/test_search_contract.py):

    * at most `k` results, best first (scores never increase down the list);
    * ties are ordered by document position, so results are deterministic;
    * no range appears twice;
    * a blank query returns no results, `k < 1` raises ValueError;
    * scores are finite numbers.
    """

    name: str

    def search(self, query: str, k: int = 10) -> list[SearchResult]: ...


_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens: 'BM25-style Retrieval!' -> ['bm25', 'style', 'retrieval']."""
    return _TOKEN.findall(text.lower())


def check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def top_k(scored: Iterable[tuple[float, Chunk]], k: int) -> list[SearchResult]:
    """Turn (score, chunk) pairs into the k best SearchResults, following the contract.

    Every chunk-based backend ends with this, so ordering and tie-breaking are the same
    everywhere. Zero and negative scores mean "no match" and are dropped.
    """
    check_k(k)
    ranked = sorted(
        ((s, c) for s, c in scored if s > 0 and math.isfinite(s)),
        key=lambda pair: (-pair[0], pair[1].doc_id, pair[1].start),
    )
    return [SearchResult.from_chunk(chunk, score) for score, chunk in ranked[:k]]


class KeywordSearcher:
    """The simplest possible backend: count how often the query's words occur in a chunk.

    It is a reference implementation for the contract, and a floor for the benchmark:
    any real method should beat it.
    """

    name = "keyword"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokens = [tokenize(c.text) for c in chunks]

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        check_k(k)
        terms = set(tokenize(query))
        if not terms:
            return []
        scored = (
            (float(sum(token in terms for token in tokens)), chunk)
            for tokens, chunk in zip(self._tokens, self.chunks, strict=True)
        )
        return top_k(scored, k)
