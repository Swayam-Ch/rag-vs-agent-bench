"""Hybrid search: combine several rankings with reciprocal rank fusion (RRF).

BM25 scores (e.g. 14.7) and cosine similarities (e.g. 0.63) live on different scales, so
adding them up would mostly measure which method has bigger numbers. RRF ignores the
scores and uses only each result's *rank* in each list:

    rrf(d) = sum over rankings r that contain d of  1 / (k + rank_r(d))

with ranks starting at 1 and k = 60 (the value from the original RRF paper, Cormack
et al., 2009). A result near the top of *both* lists beats one that is first in only
one list: 1/61 + 1/62 ~ 0.0325 > 1/61 ~ 0.0164. The larger k is, the less it matters
*where* in a list a result appears, compared with *how many* lists it appears in.

Results are matched across rankings by position in the corpus (document + character
range), so all fused methods must run on the same chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ragbench.search.base import Searcher, SearchResult, check_k

RRF_K = 60
DEFAULT_DEPTH = 50  # how many results to take from each method before fusing


def _key(result: SearchResult) -> tuple[str, int, int]:
    return (result.doc_id, result.start, result.end)


def rrf(rankings: Sequence[Sequence[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
    """Fuse ranked lists into one, best first. Each result's score is its fused RRF score.

    Ties are broken by document position, like every other method.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    fused: dict[tuple[str, int, int], float] = {}
    first_seen: dict[tuple[str, int, int], SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            key = _key(result)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, result)
    order = sorted(fused, key=lambda key: (-fused[key], key))
    return [replace(first_seen[key], score=fused[key]) for key in order]


class HybridSearcher:
    """Runs several searchers on the same query and fuses their rankings with RRF.

    Each searcher is asked for `max(depth, k)` results. So for k <= depth, a smaller k
    only truncates the list; asking for more than `depth` results looks deeper into
    each ranking, which can add RRF points to some results and reorder the top. The
    benchmark uses one fixed k, so results stay comparable across methods.
    """

    def __init__(
        self,
        searchers: Sequence[Searcher],
        depth: int = DEFAULT_DEPTH,
        rrf_k: int = RRF_K,
        name: str = "hybrid",
    ):
        if not searchers:
            raise ValueError("need at least one searcher to fuse")
        if depth < 1 or rrf_k < 0:
            raise ValueError(f"need depth >= 1 and rrf_k >= 0, got {depth}, {rrf_k}")
        self.searchers = list(searchers)
        self.depth, self.rrf_k, self.name = depth, rrf_k, name

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        check_k(k)
        if not query.strip():
            return []
        # look at least k deep in each list, so asking for more results never hides any
        depth = max(self.depth, k)
        rankings = [searcher.search(query, k=depth) for searcher in self.searchers]
        return rrf(rankings, k=self.rrf_k)[:k]
