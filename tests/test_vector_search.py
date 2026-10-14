import os

import numpy as np
import pytest
from fakes import HashEmbedder

from ragbench.pipeline.chunk import Chunk
from ragbench.search.vector_search import VectorSearcher, fingerprint, normalize

TEXTS = [
    "The Transformer relies entirely on attention mechanisms.",
    "Dense passage retrieval encodes questions and passages into vectors.",
    "BM25 is a strong lexical retrieval baseline.",
    "We evaluate on open-domain question answering benchmarks.",
]


def _chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(f"d{i}", f"d{i}:0", 0, len(t), 1, 1, t) for i, t in enumerate(texts)]


CHUNKS = _chunks(TEXTS)


# --------------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------------- #


def test_normalize_gives_unit_rows():
    rows = normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)
    assert np.allclose(rows[0], [0.6, 0.8])


def test_normalize_leaves_zero_rows_at_zero_without_nan():
    rows = normalize(np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert not np.isnan(rows).any()
    assert rows[0].tolist() == [0.0, 0.0]


def test_normalize_accepts_a_single_vector():
    assert normalize(np.array([0.0, 5.0])).shape == (1, 2)


def test_identical_text_scores_one_and_ranks_first():
    [best] = VectorSearcher(CHUNKS, HashEmbedder()).search(TEXTS[2], k=1)
    assert best.doc_id == "d2"
    assert best.score == pytest.approx(1.0, abs=1e-6)


def test_vector_length_does_not_matter_only_direction():
    small = VectorSearcher(CHUNKS, HashEmbedder(scale=1.0)).search("passage retrieval", k=4)
    large = VectorSearcher(CHUNKS, HashEmbedder(scale=1000.0)).search("passage retrieval", k=4)
    assert [r.doc_id for r in small] == [r.doc_id for r in large]
    assert [r.score for r in small] == pytest.approx([r.score for r in large], rel=1e-5)


def test_scores_are_cosines_between_minus_one_and_one():
    for r in VectorSearcher(CHUNKS, HashEmbedder()).search("retrieval baseline", k=4):
        assert -1.0 - 1e-6 <= r.score <= 1.0 + 1e-6


def test_empty_corpus_and_blank_query():
    assert VectorSearcher([], HashEmbedder()).search("attention", k=3) == []
    assert VectorSearcher(CHUNKS, HashEmbedder()).search("  ", k=3) == []


# --------------------------------------------------------------------------- #
# The embedding cache
# --------------------------------------------------------------------------- #


def test_second_run_loads_from_cache_without_embedding_the_corpus(tmp_path):
    first = HashEmbedder()
    VectorSearcher(CHUNKS, first, cache_dir=tmp_path)
    assert first.calls == [TEXTS]  # the whole corpus, once
    assert len(list(tmp_path.glob("embeddings.*.npy"))) == 1

    second = HashEmbedder()
    searcher = VectorSearcher(CHUNKS, second, cache_dir=tmp_path)
    assert second.calls == []  # nothing embedded...
    searcher.search("attention", k=1)
    assert second.calls == [["attention"]]  # ...except the query


def test_cached_and_fresh_results_are_identical(tmp_path):
    fresh = VectorSearcher(CHUNKS, HashEmbedder()).search("question answering", k=4)
    VectorSearcher(CHUNKS, HashEmbedder(), cache_dir=tmp_path)
    cached = VectorSearcher(CHUNKS, HashEmbedder(), cache_dir=tmp_path).search(
        "question answering", k=4
    )
    assert cached == fresh


@pytest.mark.parametrize("change", ["text", "model", "chunk_id"])
def test_any_change_invalidates_the_cache(tmp_path, change):
    VectorSearcher(CHUNKS, HashEmbedder(), cache_dir=tmp_path)
    chunks, name = list(CHUNKS), "hash-256"
    if change == "text":
        c = chunks[0]
        chunks[0] = Chunk(c.doc_id, c.chunk_id, c.start, c.end, 1, 1, c.text + "!")
    elif change == "model":
        name = "another-model"
    else:
        c = chunks[0]
        chunks[0] = Chunk(c.doc_id, "other:0", c.start, c.end, 1, 1, c.text)

    embedder = HashEmbedder(name=name)
    VectorSearcher(chunks, embedder, cache_dir=tmp_path)
    assert len(embedder.calls) == 1  # recomputed, not reused
    assert len(list(tmp_path.glob("embeddings.*.npy"))) == 2


def test_fingerprint_is_stable_and_sensitive():
    assert fingerprint("m", CHUNKS) == fingerprint("m", list(CHUNKS))
    assert fingerprint("m", CHUNKS) != fingerprint("m", CHUNKS[:-1])
    assert fingerprint("m", CHUNKS) != fingerprint("n", CHUNKS)


def test_cache_with_wrong_number_of_rows_is_recomputed(tmp_path):
    path = tmp_path / f"embeddings.{fingerprint('hash-256', CHUNKS)}.npy"
    np.save(path, np.zeros((2, 256), dtype=np.float32))  # stale/corrupt: 2 rows, 4 chunks
    embedder = HashEmbedder()
    VectorSearcher(CHUNKS, embedder, cache_dir=tmp_path)
    assert len(embedder.calls) == 1
    assert np.load(path).shape == (4, 256)


def test_no_cache_dir_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    VectorSearcher(CHUNKS, HashEmbedder())
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# The real model (opt-in: downloads ~80 MB the first time)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("RAGBENCH_REAL_MODEL"), reason="set RAGBENCH_REAL_MODEL=1 to run"
)
def test_real_model_matches_meaning_not_words():
    from ragbench.search.vector_search import SentenceTransformerEmbedder

    texts = [
        "The cat sat on the mat.",
        "Dense retrieval maps questions and passages into the same vector space.",
        "Stock prices fell sharply on Monday.",
    ]
    searcher = VectorSearcher(_chunks(texts), SentenceTransformerEmbedder())
    # no word in common with the right answer except "and"
    [best] = searcher.search("embedding queries and documents for semantic search", k=1)
    assert best.doc_id == "d1"


# --------------------------------------------------------------------------- #
# Model version pinning
# --------------------------------------------------------------------------- #


def test_model_revision_is_pinned_and_part_of_the_fingerprint(monkeypatch):
    import sys
    import types

    from ragbench.search import vector_search

    calls = []

    class FakeSentenceTransformer:
        def __init__(self, name, revision=None):
            calls.append((name, revision))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    pinned = vector_search.SentenceTransformerEmbedder()
    other = vector_search.SentenceTransformerEmbedder(revision="abc123")

    assert calls[0] == (vector_search.DEFAULT_MODEL, vector_search.DEFAULT_REVISION)
    assert len(vector_search.DEFAULT_REVISION) == 40  # a full commit hash, not "main"
    assert pinned.name.endswith("@" + vector_search.DEFAULT_REVISION)
    assert fingerprint(pinned.name, CHUNKS) != fingerprint(other.name, CHUNKS)
