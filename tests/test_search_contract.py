"""The Searcher contract, checked for every backend.

To test a new backend, add a factory to SEARCHERS: it receives the corpus chunks and
returns a ready-to-query searcher. Every check below then runs against it too.
"""

import math
from collections.abc import Callable

import pytest

from ragbench.pipeline.chunk import Chunk, chunk_paragraphs
from ragbench.search.base import (
    KeywordSearcher,
    Searcher,
    SearchResult,
    check_k,
    tokenize,
    top_k,
)
from ragbench.search.bm25 import BM25Searcher

DOCS = {
    "attention": "The Transformer relies entirely on attention.\n\n"
    "Multi-head attention lets the model attend to several positions.\n\n"
    "We train on WMT 2014 English-German.",
    "rag": "Retrieval-augmented generation combines a retriever with a generator.\n\n"
    "Dense passage retrieval encodes questions and passages.\n\n"
    "RAG improves open-domain question answering.",
    "bm25": "BM25 is a lexical retrieval baseline.\n\n"
    "It scores documents by term frequency and inverse document frequency.",
}

QUERIES = ["attention", "dense retrieval", "question answering", "BM25 baseline", "zebra"]


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id, text in DOCS.items():
        out += chunk_paragraphs(doc_id, text, [0], max_size=80, min_size=0)
    return out


SEARCHERS: list[Callable[[list[Chunk]], Searcher]] = [
    KeywordSearcher,
    BM25Searcher,
]


def assert_follows_contract(searcher: Searcher, chunks: list[Chunk]) -> None:
    """Every rule from the Searcher docstring, as assertions."""
    assert isinstance(searcher, Searcher)
    assert isinstance(searcher.name, str) and searcher.name
    texts = {c.doc_id: DOCS[c.doc_id] for c in chunks}

    for query in QUERIES:
        for k in (1, 3, 100):
            results = searcher.search(query, k=k)
            assert len(results) <= k
            assert all(isinstance(r, SearchResult) for r in results)
            scores = [r.score for r in results]
            assert all(math.isfinite(s) for s in scores)
            assert scores == sorted(scores, reverse=True), "best first"
            ranges = [(r.doc_id, r.start, r.end) for r in results]
            assert len(set(ranges)) == len(ranges), "no range twice"
            for r in results:  # every result points at real text
                assert texts[r.doc_id][r.start : r.end] == r.text
                assert 1 <= r.page <= r.page_end
        assert searcher.search(query, k=5) == searcher.search(query, k=5), "deterministic"
        assert searcher.search(query, k=1) == searcher.search(query, k=5)[:1], "k only truncates"

    assert searcher.search("", k=5) == []
    assert searcher.search("   \n", k=5) == []
    with pytest.raises(ValueError):
        searcher.search("attention", k=0)


@pytest.mark.parametrize("factory", SEARCHERS, ids=lambda f: f.__name__)
def test_searcher_follows_contract(factory, chunks):
    assert_follows_contract(factory(chunks), chunks)


@pytest.mark.parametrize("factory", SEARCHERS, ids=lambda f: f.__name__)
def test_obvious_query_finds_the_right_document(factory, chunks):
    results = factory(chunks).search("multi-head attention", k=1)
    assert results and results[0].doc_id == "attention"


# --------------------------------------------------------------------------- #
# The contract checker must catch broken searchers, or it proves nothing
# --------------------------------------------------------------------------- #


class _Broken:
    name = "broken"

    def __init__(self, chunks, flaw):
        self.chunks, self.flaw = chunks, flaw

    def search(self, query, k=10):
        check_k(k)
        if not query.strip():
            return []
        results = [SearchResult.from_chunk(c, float(i)) for i, c in enumerate(self.chunks)]
        results.sort(key=lambda r: -r.score)
        if self.flaw == "worst_first":
            results.reverse()
        if self.flaw == "duplicates":
            results = [results[0], results[0]]
        if self.flaw == "too_many":
            return results
        if self.flaw == "wrong_text":
            results = [SearchResult(**{**r.__dict__, "text": "made up"}) for r in results]
        return results[:k]


@pytest.mark.parametrize("flaw", ["worst_first", "duplicates", "too_many", "wrong_text"])
def test_contract_catches_broken_searchers(chunks, flaw):
    with pytest.raises(AssertionError):
        assert_follows_contract(_Broken(chunks, flaw), chunks)


def test_contract_passes_the_unbroken_version(chunks):
    assert_follows_contract(_Broken(chunks, flaw=None), chunks)


# --------------------------------------------------------------------------- #
# Helpers and KeywordSearcher specifics
# --------------------------------------------------------------------------- #


def test_tokenize():
    assert tokenize("BM25-style Retrieval!") == ["bm25", "style", "retrieval"]
    assert tokenize("  ") == []


def test_top_k_breaks_ties_by_document_position():
    a = Chunk("b", "b:0", 50, 60, 1, 1, "x")
    b = Chunk("a", "a:1", 90, 99, 1, 1, "y")
    c = Chunk("a", "a:0", 10, 20, 1, 1, "z")
    assert [r.chunk_id for r in top_k([(1.0, a), (1.0, b), (1.0, c)], k=3)] == [
        "a:0",
        "a:1",
        "b:0",
    ]


def test_top_k_drops_non_matches():
    c = Chunk("a", "a:0", 0, 1, 1, 1, "x")
    assert top_k([(0.0, c), (-1.0, c), (float("nan"), c)], k=5) == []


def test_keyword_score_is_the_number_of_matching_words(chunks):
    searcher = KeywordSearcher(chunks)
    [best] = searcher.search("multi head attention", k=1)
    assert best.text.startswith("Multi-head attention")
    assert best.score == 3.0  # multi + head + attention


def test_repeating_a_query_word_does_not_count_twice(chunks):
    searcher = KeywordSearcher(chunks)
    assert searcher.search("attention attention", k=5) == searcher.search("attention", k=5)


def test_keyword_search_ignores_case_and_punctuation(chunks):
    [best] = KeywordSearcher(chunks).search("BM25!", k=1)
    assert best.doc_id == "bm25"
