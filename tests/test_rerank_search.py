"""Reranking a first stage's top candidates with a pair scorer (a cross-encoder)."""

import pytest

from ragbench.pipeline.chunk import chunk_paragraphs
from ragbench.search.base import SearchResult
from ragbench.search.rerank_search import RerankSearcher
from ragbench.search.vector_search import VectorSearcher, resolve_revision


def _result(doc, start, score, text="x"):
    return SearchResult(doc, start, start + len(text), 1, 1, score, text)


class FixedBase:
    name = "base"

    def __init__(self, results):
        self.results, self.asked = results, []

    def search(self, query, k=10):
        self.asked.append(k)
        return self.results[:k]


class TableScorer:
    name = "table"

    def __init__(self, scores):
        self.scores = scores

    def score(self, query, texts):
        return [self.scores[t] for t in texts]


BASE = [_result("a", 0, 9.0, "one"), _result("a", 10, 8.0, "two"), _result("b", 0, 7.0, "three")]


def test_reorders_candidates_by_the_scorer_and_keeps_its_scores():
    scorer = TableScorer({"one": -1.0, "two": 3.5, "three": 0.2})  # logits can be negative
    out = RerankSearcher(FixedBase(BASE), scorer, depth=3).search("q", k=3)
    assert [(r.text, r.score) for r in out] == [("two", 3.5), ("three", 0.2), ("one", -1.0)]


def test_always_reranks_the_same_depth_so_k_only_truncates():
    base = FixedBase(BASE)
    searcher = RerankSearcher(base, TableScorer({"one": 1, "two": 3, "three": 2}), depth=3)
    top1, top3 = searcher.search("q", k=1), searcher.search("q", k=3)
    assert top1 == top3[:1] and base.asked == [3, 3]
    assert len(searcher.search("q", k=10)) == 3  # never more than depth


def test_ties_are_ordered_by_document_position():
    scorer = TableScorer({"one": 1.0, "two": 1.0, "three": 1.0})
    out = RerankSearcher(FixedBase(BASE[::-1]), scorer, depth=3).search("q", k=3)
    assert [(r.doc_id, r.start) for r in out] == [("a", 0), ("a", 10), ("b", 0)]


def test_non_finite_scores_sink_to_the_bottom():
    scorer = TableScorer({"one": float("nan"), "two": 2.0, "three": -3.0})
    out = RerankSearcher(FixedBase(BASE), scorer, depth=3).search("q", k=3)
    assert [r.text for r in out] == ["two", "three", "one"]
    assert out[-1].score == -3.0  # replaced by the lowest finite score, still best-first


def test_name_and_validation():
    assert RerankSearcher(FixedBase(BASE), TableScorer({})).name == "base+rerank"
    with pytest.raises(ValueError):
        RerankSearcher(FixedBase(BASE), TableScorer({}), depth=0)


# --------------------------------------------------------------------------- #
# Query and document prefixes; revision resolving
# --------------------------------------------------------------------------- #


class PrefixRecorder:
    name = "rec"

    def __init__(self):
        self.docs, self.queries = [], []

    def embed(self, texts):
        import numpy as np

        self.docs += texts
        return np.ones((len(texts), 2), dtype="float32")

    def embed_query(self, texts):
        import numpy as np

        self.queries += texts
        return np.ones((len(texts), 2), dtype="float32")


def test_vector_search_uses_embed_query_for_queries_when_the_model_has_it():
    chunks = chunk_paragraphs("d", "Some text.", [0], max_size=100, min_size=0)
    rec = PrefixRecorder()
    VectorSearcher(chunks, rec).search("hello", k=1)
    assert rec.docs == ["Some text."] and rec.queries == ["hello"]


def test_explicit_revision_is_used_as_is():
    assert resolve_revision("any/model", "abc123") == "abc123"


def test_missing_revision_is_looked_up(monkeypatch):
    import types

    fake = types.ModuleType("huggingface_hub")
    fake.model_info = lambda name: types.SimpleNamespace(sha=f"sha-of-{name}")
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake)
    assert resolve_revision("BAAI/bge-base-en-v1.5", None) == "sha-of-BAAI/bge-base-en-v1.5"


def test_lookup_failure_says_what_to_do(monkeypatch):
    import types

    fake = types.ModuleType("huggingface_hub")

    def offline(name):
        raise OSError("no network")

    fake.model_info = offline
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake)
    with pytest.raises(RuntimeError, match="pass an explicit revision"):
        resolve_revision("x/y", None)
