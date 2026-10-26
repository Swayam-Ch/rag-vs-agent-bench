"""Retrieval metrics, checked against values worked out by hand.

The main fixture, worked through once so every expected number below can be checked:

    gold:  A p.3            and   B p.5 or p.7
    rank   result           hits gold (page level)?     hits gold paper?
     1     C p.3            no  (wrong paper)           no
     2     A p.1-2          no  (right paper, p.3 not)  yes, A
     3     B p.6-7          yes, B (7 is in 6-7)        yes, B
     4     A p.2-3          yes, A (3 is in 2-3)        yes, A
     5     A p.3            yes, A again (counts once)  yes, A again

    page level:  recall@1 = 0/2, @2 = 0/2, @3 = 1/2, @4 = 2/2;  first hit at rank 3 -> RR 1/3
    paper level: recall@1 = 0/2, @2 = 1/2, @3 = 2/2;            first hit at rank 2 -> RR 1/2
"""

import pytest

from ragbench.eval.metrics import (
    RETRIEVAL_TYPES,
    Hit,
    hits_from_results,
    mean,
    mrr,
    page_hit,
    recall_at_k,
    reciprocal_rank,
)
from ragbench.eval.questions import TYPES, Evidence
from ragbench.search.base import SearchResult

GOLD = [Evidence("A", (3,)), Evidence("B", (5, 7))]
HITS = [Hit("C", 3, 3), Hit("A", 1, 2), Hit("B", 6, 7), Hit("A", 2, 3), Hit("A", 3, 3)]


# --------------------------------------------------------------------------- #
# page_hit: one result against one gold entry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("hit", "evidence", "expected"),
    [
        (Hit("A", 3, 3), Evidence("A", (3,)), True),  # exactly the gold page
        (Hit("A", 2, 3), Evidence("A", (3,)), True),  # gold page at the END of the range
        (Hit("A", 3, 4), Evidence("A", (3,)), True),  # gold page at the start
        (Hit("A", 1, 9), Evidence("A", (5,)), True),  # inside a long range
        (Hit("A", 4, 5), Evidence("A", (3,)), False),  # just after
        (Hit("A", 1, 2), Evidence("A", (3,)), False),  # just before
        (Hit("B", 3, 3), Evidence("A", (3,)), False),  # right page, wrong paper
        (Hit("A", 7, 7), Evidence("A", (5, 7)), True),  # any one of the pages is enough
        (Hit("A", 9, 9), Evidence("A", ()), True),  # no pages (negation): any page of A
        (Hit("B", 9, 9), Evidence("A", ()), False),
        # the design note on #16: a grep snippet whose leading context starts on p.4
        # while the match is on p.5 must count for gold p.5
        (Hit("A", 4, 5), Evidence("A", (5,)), True),
    ],
)
def test_page_hit(hit, evidence, expected):
    assert page_hit(hit, evidence) is expected


# --------------------------------------------------------------------------- #
# recall@k and reciprocal rank on the worked example
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("k", "level", "expected"),
    [
        (1, "page", 0.0),
        (2, "page", 0.0),
        (3, "page", 0.5),
        (4, "page", 1.0),
        (5, "page", 1.0),  # the duplicate A p.3 doesn't push recall past 1
        (100, "page", 1.0),  # k beyond the list is fine
        (None, "page", 1.0),  # None: all results (the agent's evidence has no fixed k)
        (1, "paper", 0.0),
        (2, "paper", 0.5),
        (3, "paper", 1.0),
    ],
)
def test_recall_at_k(k, level, expected):
    assert recall_at_k(HITS, GOLD, k, level) == pytest.approx(expected)


@pytest.mark.parametrize(("level", "expected"), [("page", 1 / 3), ("paper", 1 / 2)])
def test_reciprocal_rank(level, expected):
    assert reciprocal_rank(HITS, GOLD, level) == pytest.approx(expected)


def test_nothing_found_scores_zero():
    misses = [Hit("C", 1, 1), Hit("A", 9, 9)]
    assert recall_at_k(misses, GOLD, 10) == 0.0
    assert reciprocal_rank(misses, GOLD) == 0.0
    assert recall_at_k([], GOLD, 10) == 0.0
    assert reciprocal_rank([], GOLD) == 0.0


def test_first_result_right_scores_one():
    assert reciprocal_rank([Hit("A", 3, 3), Hit("C", 1, 1)], GOLD) == 1.0


def test_three_papers_two_found():
    gold = [Evidence("A", (1,)), Evidence("B", (1,)), Evidence("C", (1,))]
    hits = [Hit("C", 1, 1), Hit("X", 1, 1), Hit("A", 1, 1)]
    assert recall_at_k(hits, gold, 2) == pytest.approx(1 / 3)
    assert recall_at_k(hits, gold, 3) == pytest.approx(2 / 3)


def test_same_paper_listed_twice_needs_both_pages_but_counts_once_as_a_paper():
    # a question needing two separate facts from one paper lists it twice
    gold = [Evidence("A", (2,)), Evidence("A", (8,))]
    hits = [Hit("A", 2, 2)]
    assert recall_at_k(hits, gold, level="page") == 0.5  # one of the two facts
    assert recall_at_k(hits, gold, level="paper") == 1.0  # the one paper was found


# --------------------------------------------------------------------------- #
# Undefined cases: no gold (unanswerable) -> None, never a misleading 0
# --------------------------------------------------------------------------- #


def test_no_gold_is_undefined_not_zero():
    assert recall_at_k(HITS, [], 5) is None
    assert reciprocal_rank(HITS, []) is None


def test_mean_skips_undefined():
    assert mean([1.0, None, 0.0, 0.5]) == pytest.approx(0.5)
    assert mean([None, None]) is None
    assert mean([]) is None


def test_mrr_over_questions():
    # per question: rank 1 -> 1, rank 3 -> 1/3, not found -> 0, no gold -> skipped
    ranked = [
        [Hit("A", 3, 3)],
        [Hit("X", 1, 1), Hit("X", 2, 2), Hit("A", 3, 3)],
        [Hit("X", 1, 1)],
        [Hit("A", 3, 3)],
    ]
    golds = [[Evidence("A", (3,))]] * 3 + [[]]
    assert mrr(ranked, golds) == pytest.approx((1 + 1 / 3 + 0) / 3)  # = 4/9


# --------------------------------------------------------------------------- #
# Bad input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(HITS, GOLD, k)


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="level must be one of"):
        recall_at_k(HITS, GOLD, 5, level="doc")


def test_mrr_needs_one_result_list_per_question():
    with pytest.raises(ValueError, match="2 result lists for 1 questions"):
        mrr([[], []], [GOLD])


# --------------------------------------------------------------------------- #
# From search results to hits
# --------------------------------------------------------------------------- #


def _result(doc_id, page, page_end):
    return SearchResult(doc_id, 0, 10, page, page_end, 1.0, "text")


def test_hits_from_results_maps_doc_ids_to_arxiv_ids():
    results = [_result("7407131fd633", 4, 5), _result("unknown", 1, 1)]
    hits = hits_from_results(results, {"7407131fd633": "2005.11401"})
    assert hits == [Hit("2005.11401", 4, 5), Hit("unknown", 1, 1)]
    # an unmapped document matches no gold paper instead of crashing
    assert recall_at_k(hits[1:], [Evidence("2005.11401", (1,))]) == 0.0


def test_retrieval_types_are_question_types_without_negation_and_unanswerable():
    assert set(RETRIEVAL_TYPES) == set(TYPES) - {"negation", "unanswerable"}
