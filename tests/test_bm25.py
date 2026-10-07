import math

import pytest

from ragbench.pipeline.chunk import Chunk
from ragbench.search.base import tokenize
from ragbench.search.bm25 import BM25Searcher

TEXTS = [
    "The Transformer relies entirely on attention mechanisms.",
    "Multi-head attention attends to several positions at once.",
    "Dense passage retrieval encodes questions and passages into vectors.",
    "BM25 is a strong lexical retrieval baseline for passage ranking.",
    "Retrieval-augmented generation combines a retriever and a generator.",
    "We evaluate on open-domain question answering benchmarks.",
    "A coin is heads up. Heads, heads, heads: the coin is still heads up.",
    "Attention heads in the encoder learn different relations.",
]


def _chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(f"d{i}", f"d{i}:0", 0, len(t), 1, 1, t) for i, t in enumerate(texts)]


CHUNKS = _chunks(TEXTS)


# --------------------------------------------------------------------------- #
# Cross-check against the rank_bm25 library
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    ["attention", "passage retrieval", "multi head attention", "question answering", "coin"],
)
def test_okapi_scores_match_rank_bm25_exactly(query):
    rank_bm25 = pytest.importorskip("rank_bm25")
    reference = rank_bm25.BM25Okapi([tokenize(t) for t in TEXTS], k1=1.5, b=0.75)
    expected = reference.get_scores(tokenize(query))

    ours = BM25Searcher(CHUNKS, k1=1.5, b=0.75, idf="okapi").scores(query)

    for i, want in enumerate(expected):
        assert ours.get(i, 0.0) == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("query", ["attention heads", "retrieval passage baseline"])
def test_ranking_matches_rank_bm25(query):
    rank_bm25 = pytest.importorskip("rank_bm25")
    reference = rank_bm25.BM25Okapi([tokenize(t) for t in TEXTS])
    ref_scores = reference.get_scores(tokenize(query))
    matching = [i for i in range(len(TEXTS)) if ref_scores[i] > 0]
    ref_order = [f"d{i}" for i in sorted(matching, key=lambda i: (-ref_scores[i], i))]

    ours = [r.doc_id for r in BM25Searcher(CHUNKS, idf="okapi").search(query, k=len(TEXTS))]

    assert ours == ref_order


# --------------------------------------------------------------------------- #
# Each part of the formula does its job
# --------------------------------------------------------------------------- #


def test_saturation_repetition_cannot_beat_several_matching_words():
    # the keyword baseline's failure on the real corpus: "heads" x5 in a coin-flip example
    searcher = BM25Searcher(CHUNKS)
    [best] = searcher.search("attention heads encoder", k=1)
    assert best.text.startswith("Attention heads in the encoder")


def test_a_word_contributes_at_most_k1_plus_1_times_its_idf():
    searcher = BM25Searcher(_chunks(["heads " * 1000]), k1=1.5, b=0.0)
    ceiling = (1.5 + 1) * searcher.idf["heads"]
    assert searcher.scores("heads")[0] < ceiling
    assert searcher.scores("heads")[0] > 0.99 * ceiling  # ...and gets close with enough repeats


def test_shorter_chunk_wins_with_equal_term_frequency():
    texts = ["retrieval works", "retrieval works " + "filler " * 30, "unrelated text"]
    scores = BM25Searcher(_chunks(texts), b=0.75).scores("retrieval")
    assert scores[0] > scores[1]


def test_b_zero_turns_length_normalisation_off():
    texts = ["retrieval works", "retrieval works " + "filler " * 30, "unrelated text"]
    scores = BM25Searcher(_chunks(texts), b=0.0).scores("retrieval")
    assert scores[0] == pytest.approx(scores[1])


def test_rare_words_weigh_more_than_common_ones():
    searcher = BM25Searcher(CHUNKS)
    # chunks containing the word:   coin 1,  passage 2,  attention 3
    assert [len(searcher.postings[w]) for w in ("coin", "passage", "attention")] == [1, 2, 3]
    assert searcher.idf["coin"] > searcher.idf["passage"] > searcher.idf["attention"]


def test_lucene_idf_is_positive_even_for_a_word_in_every_chunk():
    searcher = BM25Searcher(_chunks(["the a", "the b", "the c"]), idf="lucene")
    assert searcher.idf["the"] > 0


def test_okapi_idf_floors_words_in_most_chunks():
    # "the" is in 3 of 4 chunks: raw okapi idf is negative, so it gets the floor
    searcher = BM25Searcher(_chunks(["the a", "the b", "the c", "d e f"]), idf="okapi")
    raw = math.log((4 - 3 + 0.5) / (3 + 0.5))
    assert raw < 0
    others = [math.log((4 - 1 + 0.5) / (1 + 0.5))] * 6  # a, b, c, d, e, f: one chunk each
    floor = 0.25 * (raw + sum(others)) / 7
    assert searcher.idf["the"] == pytest.approx(floor) and floor > 0


def test_okapi_floor_can_itself_be_negative_like_rank_bm25():
    """A quirk kept for exactness: the floor is 0.25 x the *average* idf, which is negative
    when most words are common. The default "lucene" idf has no such problem."""
    searcher = BM25Searcher(_chunks(["the a", "the b", "the c"]), idf="okapi")
    assert searcher.idf["the"] < 0
    rank_bm25 = pytest.importorskip("rank_bm25")
    reference = rank_bm25.BM25Okapi([["the", "a"], ["the", "b"], ["the", "c"]])
    assert searcher.idf["the"] == pytest.approx(reference.idf["the"])


def test_unknown_words_and_repeats():
    searcher = BM25Searcher(CHUNKS)
    assert searcher.search("zebra", k=5) == []
    assert searcher.search("attention attention", k=5) == searcher.search("attention", k=5)


def test_only_chunks_containing_a_query_word_are_scored():
    assert set(BM25Searcher(CHUNKS).scores("coin")) == {6}


@pytest.mark.parametrize("kwargs", [{"idf": "tfidf"}, {"k1": -1.0}, {"b": 1.5}, {"b": -0.1}])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        BM25Searcher(CHUNKS, **kwargs)


def test_empty_corpus_returns_nothing():
    assert BM25Searcher([]).search("attention", k=3) == []
