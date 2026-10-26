"""Retrieval metrics: did a method find the evidence, and how high did it rank it?

These score *retrieval* only, separately from whether the final answer is right (that is
the judge's job, #18). A method can retrieve the right page and still answer wrongly, or
answer correctly from memory without retrieving anything; keeping the two scores apart
shows which happened.

The unit of evidence is one `gold` entry of a question: a paper plus the pages where the
evidence is. **Any one of an entry's pages is enough** to count it as found (the pages
are alternatives, e.g. a fact stated in the body and repeated in an appendix). A
question that needs two separate facts from the same paper lists that paper twice.

A result covers a page *range* `[page, page_end]`: a chunk or a grep snippet can start
at the bottom of one page and end at the top of the next. A gold page counts as found if
it lies anywhere in that range, not only at the start.

Two levels:

* ``"page"``: the result must be from the right paper AND cover one of the gold pages.
* ``"paper"``: any result from the right paper counts. Looser; for "which papers ..."
  questions this is the headline number.

Metrics are only defined for questions with evidence to find: single_fact, multi_filter
and cross_doc (`RETRIEVAL_TYPES`). An unanswerable question has nothing to retrieve, and
a negation question's answer is the papers that *lack* something, which no passage
shows; both are scored on the answer alone. The functions return None for an empty
gold list so a caller can't average a meaningless 0 into the results by accident.

All functions are pure: plain data in, a number out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ragbench.eval.questions import Evidence
from ragbench.search.base import SearchResult

Level = Literal["page", "paper"]
LEVELS: tuple[Level, ...] = ("page", "paper")
RETRIEVAL_TYPES = ("single_fact", "multi_filter", "cross_doc")


@dataclass(frozen=True)
class Hit:
    """One retrieved result, reduced to what the metrics need."""

    paper: str  # arXiv ID, the same IDs the question set uses
    page: int  # first page the result covers
    page_end: int  # last page it covers (== page for a result on one page)


def hits_from_results(results: Iterable[SearchResult], arxiv_of: Mapping[str, str]) -> list[Hit]:
    """Convert search results (which use internal doc IDs) into Hits (arXiv IDs).

    `arxiv_of` maps doc_id -> arXiv ID, from the manifest. A doc_id missing from it is
    kept as is: it then matches no gold paper, which is the honest outcome.
    """
    return [Hit(arxiv_of.get(r.doc_id, r.doc_id), r.page, r.page_end) for r in results]


def page_hit(hit: Hit, evidence: Evidence) -> bool:
    """Does this result cover one of the evidence's gold pages (in the right paper)?

    An entry without pages (allowed for negation questions: "the whole paper") is hit
    by any result from its paper.
    """
    if hit.paper != evidence.paper:
        return False
    if not evidence.pages:
        return True
    return any(hit.page <= p <= hit.page_end for p in evidence.pages)


def _units(gold: Sequence[Evidence], level: Level) -> list[Evidence]:
    """What has to be found: every gold entry (page level) or every gold paper."""
    if level == "page":
        return list(gold)
    if level == "paper":
        papers = dict.fromkeys(e.paper for e in gold)  # each paper once, in order
        return [Evidence(p, ()) for p in papers]
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def _check_k(k: int | None) -> None:
    if k is not None and k < 1:
        raise ValueError(f"k must be at least 1 (or None for all results), got {k}")


def recall_at_k(
    hits: Sequence[Hit], gold: Sequence[Evidence], k: int | None = None, level: Level = "page"
) -> float | None:
    """Fraction of the gold evidence found in the first `k` results (all if k is None).

    Several results landing on the same gold page count once. None if there is no gold
    evidence (nothing to recall).
    """
    _check_k(k)
    units = _units(gold, level)
    if not units:
        return None
    top = hits if k is None else hits[:k]
    found = sum(1 for unit in units if any(page_hit(h, unit) for h in top))
    return found / len(units)


def reciprocal_rank(
    hits: Sequence[Hit], gold: Sequence[Evidence], level: Level = "page"
) -> float | None:
    """1 / rank of the first result that hits any gold evidence; 0.0 if none does.

    1.0 means the very first result was right, 0.5 the second, and so on. None if there
    is no gold evidence.
    """
    units = _units(gold, level)
    if not units:
        return None
    for rank, h in enumerate(hits, start=1):
        if any(page_hit(h, unit) for unit in units):
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    """Average, skipping None (questions where the metric is undefined). None if empty."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def mrr(
    ranked: Sequence[Sequence[Hit]], golds: Sequence[Sequence[Evidence]], level: Level = "page"
) -> float | None:
    """Mean reciprocal rank over questions: `ranked[i]` are the hits for `golds[i]`."""
    if len(ranked) != len(golds):
        raise ValueError(f"{len(ranked)} result lists for {len(golds)} questions")
    return mean(reciprocal_rank(h, g, level) for h, g in zip(ranked, golds, strict=True))
