import pytest
from fakes import HashEmbedder

from ragbench.pipeline.chunk import Chunk
from ragbench.search.base import SearchResult
from ragbench.search.bm25 import BM25Searcher
from ragbench.search.hybrid_search import HybridSearcher, rrf
from ragbench.search.vector_search import VectorSearcher


def _r(doc: str, score: float = 1.0) -> SearchResult:
    """A result whose identity is just the document name (one range per document)."""
    return SearchResult(doc, 0, 10, 1, 1, score, f"text of {doc}", f"{doc}:0")


def _docs(results: list[SearchResult]) -> list[str]:
    return [r.doc_id for r in results]


# --------------------------------------------------------------------------- #
# rrf: worked out by hand
# --------------------------------------------------------------------------- #


def test_result_in_both_lists_beats_one_that_tops_a_single_list():
    bm25 = [_r("a"), _r("b"), _r("c")]
    vector = [_r("d"), _r("b"), _r("a")]
    # a: 1/61 + 1/63 = 0.03227   b: 2/62 = 0.03226   d: 1/61   c: 1/63
    fused = rrf([bm25, vector])
    assert _docs(fused) == ["a", "b", "d", "c"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)
    assert fused[1].score == pytest.approx(2 / 62)


def test_ranks_1_and_3_narrowly_beat_2_and_2():
    """1/x is curved, so a first place gains more than a third place loses.

    Easy to get wrong by intuition (the author of these tests did, first time round).
    """
    fused = rrf([[_r("a"), _r("b"), _r("x")], [_r("y"), _r("b"), _r("a")]])
    assert _docs(fused)[:2] == ["a", "b"]
    assert fused[0].score - fused[1].score == pytest.approx(1 / 61 + 1 / 63 - 2 / 62)


def test_raw_scores_are_ignored_only_ranks_count():
    bm25 = [_r("a", 1000.0), _r("b", 999.0)]
    vector = [_r("b", 0.01), _r("a", 0.001)]
    assert rrf([bm25, vector])[0].score == rrf([bm25, vector])[1].score  # symmetric ranks


def test_ties_are_broken_by_document_position():
    assert _docs(rrf([[_r("z")], [_r("a")]])) == ["a", "z"]


def test_single_ranking_keeps_its_order():
    ranking = [_r("c"), _r("a"), _r("b")]
    assert _docs(rrf([ranking])) == ["c", "a", "b"]


def test_k_controls_how_much_the_position_matters():
    first_in_one = [[_r("a"), _r("x1"), _r("x2"), _r("x3")], [_r("y1")]]
    low_in_two = [[_r("x1"), _r("x2"), _r("x3"), _r("b")], [_r("y1"), _r("y2"), _r("y3"), _r("b")]]
    lists = [first_in_one[0] + low_in_two[0][3:], first_in_one[1] + low_in_two[1][1:]]
    # with k=0 the top rank dominates; with a large k, appearing twice wins
    assert _docs(rrf(lists, k=0)).index("a") < _docs(rrf(lists, k=0)).index("b")
    assert _docs(rrf(lists, k=1000)).index("b") < _docs(rrf(lists, k=1000)).index("a")


def test_same_range_from_different_methods_is_merged_not_duplicated():
    fused = rrf([[_r("a")], [_r("a")], [_r("a")]])
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(3 / 61)


def test_empty_rankings():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_negative_k_raises():
    with pytest.raises(ValueError):
        rrf([[_r("a")]], k=-1)


# --------------------------------------------------------------------------- #
# HybridSearcher
# --------------------------------------------------------------------------- #


class _Fixed:
    """A searcher that always returns the same ranking and records the k it was asked for."""

    def __init__(self, name, docs):
        self.name, self.docs, self.asked = name, docs, []

    def search(self, query, k=10):
        self.asked.append(k)
        return [_r(d) for d in self.docs][:k]


def test_hybrid_fuses_its_searchers():
    hybrid = HybridSearcher([_Fixed("bm25", ["a", "b", "c"]), _Fixed("vec", ["d", "b", "a"])])
    assert _docs(hybrid.search("anything", k=4)) == ["a", "b", "d", "c"]


def test_hybrid_reads_each_list_to_depth_and_at_least_k():
    one, two = _Fixed("one", ["a"]), _Fixed("two", ["b"])
    hybrid = HybridSearcher([one, two], depth=50)
    hybrid.search("q", k=5)
    hybrid.search("q", k=80)
    assert one.asked == [50, 80] and two.asked == [50, 80]


def test_hybrid_blank_query_and_bad_settings():
    hybrid = HybridSearcher([_Fixed("one", ["a"])])
    assert hybrid.search("  ", k=3) == []
    with pytest.raises(ValueError):
        hybrid.search("q", k=0)
    for kwargs in ({"depth": 0}, {"rrf_k": -1}):
        with pytest.raises(ValueError):
            HybridSearcher([_Fixed("one", ["a"])], **kwargs)
    with pytest.raises(ValueError):
        HybridSearcher([])


def test_bm25_plus_vector_on_real_chunks():
    texts = [
        "Multi-head attention runs several attention functions in parallel.",
        "We vary the number of attention heads and measure BLEU.",
        "BLEU+case.mixed+numrefs.1+smooth.exp+tok.intl",
    ]
    chunks = [Chunk(f"d{i}", f"d{i}:0", 0, len(t), 1, 1, t) for i, t in enumerate(texts)]
    hybrid = HybridSearcher([BM25Searcher(chunks), VectorSearcher(chunks, HashEmbedder())])
    results = hybrid.search("number of attention heads", k=3)
    assert results[0].doc_id == "d1"
    assert all(r.text == texts[int(r.doc_id[1])] for r in results)
