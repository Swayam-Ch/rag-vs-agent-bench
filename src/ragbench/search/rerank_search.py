"""Reranking: let a stronger, slower model reorder another method's top candidates.

A first-stage searcher (BM25, hybrid) is fast but scores each chunk with a simple
formula. A cross-encoder reads the query and a chunk *together* and scores how well the
chunk answers it, which is much more accurate and much slower: too slow for every chunk,
fine for the top 50.

    candidates = first_stage.search(query, k=depth)   # e.g. 50
    results    = sort candidates by cross_encoder(query, chunk), best first

Results come only from the `depth` candidates, so asking for more than `depth` returns at
most `depth`. Cross-encoder scores are logits and can be negative; unlike the first-stage
methods, a reranker keeps every candidate (a low score is a low rank, not "no match").
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

from ragbench.search.base import Searcher, SearchResult, check_k
from ragbench.search.vector_search import resolve_revision

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small (90 MB), trained on MS MARCO
DEFAULT_DEPTH = 50


class PairScorer(Protocol):
    name: str  # identifies the model, recorded in run.json

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text; higher is better."""
        ...


class CrossEncoderScorer:
    """A sentence-transformers CrossEncoder (`pip install -e ".[vector]"`)."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, revision: str | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                'reranking needs sentence-transformers: pip install -e ".[vector]"'
            ) from exc
        revision = resolve_revision(model_name, revision)
        self.name = f"{model_name}@{revision}"
        self._model = CrossEncoder(model_name, revision=revision)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class RerankSearcher:
    def __init__(
        self,
        base: Searcher,
        scorer: PairScorer,
        depth: int = DEFAULT_DEPTH,
        name: str | None = None,
    ):
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.base, self.scorer, self.depth = base, scorer, depth
        self.name = name or f"{base.name}+rerank"

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        check_k(k)
        if not query.strip():
            return []
        candidates = self.base.search(query, k=self.depth)  # the same set for every k
        scores = self.scorer.score(query, [c.text for c in candidates])
        ranked = sorted(
            (
                (s if math.isfinite(s) else -math.inf, c)
                for s, c in zip(scores, candidates, strict=True)
            ),
            key=lambda pair: (-pair[0], pair[1].doc_id, pair[1].start),
        )
        # a non-finite score sinks to the bottom with the lowest finite score
        floor = min((s for s, _ in ranked if math.isfinite(s)), default=0.0)
        return [replace(c, score=s if math.isfinite(s) else floor) for s, c in ranked[:k]]
