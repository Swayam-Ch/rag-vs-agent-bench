import pytest

from ragbench.eval.questions import (
    TYPES,
    Evidence,
    QuestionSetError,
    check_against_corpus,
    parse,
)


def _doc(*questions: str, version: int = 1) -> str:
    return f"version: {version}\nquestions:\n" + "".join(questions)


def _q(
    qid="q001",
    qtype="single_fact",
    question="How many heads?",
    answer="Eight.",
    gold='      - paper: "1706.03762"\n        pages: [5]\n',
    extra="",
) -> str:
    gold_block = f"    gold:\n{gold}" if gold is not None else ""
    return (
        f"  - id: {qid}\n    type: {qtype}\n    question: {question}\n"
        f"    answer: {answer}\n{gold_block}{extra}"
    )


TWO_PAPERS = (
    '      - paper: "2004.04906"\n        pages: [2]\n'
    '      - paper: "2005.11401"\n        pages: [4, 2, 4]\n'
)


def _problems(text: str) -> list[str]:
    with pytest.raises(QuestionSetError) as exc:
        parse(text)
    return exc.value.problems


# --------------------------------------------------------------------------- #
# Valid files
# --------------------------------------------------------------------------- #


def test_valid_question_is_loaded():
    [q] = parse(_doc(_q())).questions
    assert (q.id, q.type, q.question, q.answer) == (
        "q001",
        "single_fact",
        "How many heads?",
        "Eight.",
    )
    assert q.gold == (Evidence("1706.03762", (5,)),)
    assert q.line == 3


def test_pages_are_sorted_and_deduplicated_and_papers_listed_once():
    [q] = parse(_doc(_q(qtype="multi_filter", gold=TWO_PAPERS))).questions
    assert q.gold[1] == Evidence("2005.11401", (2, 4))
    assert q.gold_papers == ("2004.04906", "2005.11401")


def test_every_type_can_be_written():
    questions = []
    for i, qtype in enumerate(TYPES):
        gold = None if qtype == "unanswerable" else TWO_PAPERS
        extra = "    gold: []\n" if qtype == "unanswerable" else ""
        questions.append(_q(qid=f"q{i}", qtype=qtype, gold=gold, extra=extra))
    assert parse(_doc(*questions)).by_type() == dict.fromkeys(TYPES, 1)


def test_negation_may_check_a_whole_paper_with_empty_pages():
    gold = '      - paper: "1706.03762"\n        pages: []\n' + TWO_PAPERS
    [q] = parse(_doc(_q(qtype="negation", gold=gold))).questions
    assert q.gold[0] == Evidence("1706.03762", ())
    assert q.gold_papers == ("1706.03762", "2004.04906", "2005.11401")
    # and the page check against the corpus skips it instead of crashing on max(())
    assert (
        check_against_corpus(
            parse(_doc(_q(qtype="negation", gold=gold))),
            {"1706.03762", "2004.04906", "2005.11401"},
            {"1706.03762": 15, "2004.04906": 13, "2005.11401": 19},
        )
        == []
    )


def test_only_negation_may_use_empty_pages():
    empty = '      - paper: "1706.03762"\n        pages: []\n'
    for qtype in ("single_fact", "multi_filter"):
        problems = _problems(_doc(_q(qtype=qtype, gold=empty)))
        assert any("only a negation question may use []" in p for p in problems), problems


def test_line_numbers_point_at_each_question():
    text = _doc(_q(qid="q001"), _q(qid="q002"))
    assert [q.line for q in parse(text).questions] == [3, 10]  # each _q() is 7 lines


# --------------------------------------------------------------------------- #
# The issue's "done when": an unknown type fails with a clear message
# --------------------------------------------------------------------------- #


def test_unknown_type_fails_clearly_with_line_number():
    [problem] = _problems(_doc(_q(qtype="multi-filter")))
    assert problem == (
        "questions.yaml:3: q001: unknown type 'multi-filter'; expected one of: "
        "single_fact, multi_filter, cross_doc, negation, unanswerable"
    )


# --------------------------------------------------------------------------- #
# Every other mistake, and all of them reported at once
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_doc(_q(), version=2), "'version' must be 1"),
        ("version: 1\nquestions: []\n", "'questions' must be a non-empty list"),
        ("- just\n- a list\n", "expected a mapping"),
        ("version: 1\nquestions: [unclosed\n", "not valid YAML"),
        (_doc(_q()) + "extra: 1\n", "unknown top-level keys: extra"),
        (_doc(_q(extra="    pagez: [3]\n")), "unknown fields: pagez"),
        (_doc(_q(qid="Q 1")), "'id' must be lowercase"),
        (_doc(_q(question="''")), "'question' must be a non-empty text"),
        (_doc(_q(answer="''")), "'answer' must be a non-empty text"),
        (_doc(_q(gold=None)), "'gold' must list at least one paper"),
        (
            _doc(_q(gold="      - paper: 1706.03762\n        pages: [5]\n")),
            "put the arXiv ID in quotes",
        ),
        (
            _doc(_q(gold='      - paper: "attention paper"\n        pages: [5]\n')),
            "is not an arXiv ID",
        ),
        (
            _doc(_q(gold='      - paper: "1706.03762"\n        pages: []\n')),
            "non-empty list of page numbers",
        ),
        (
            _doc(_q(gold='      - paper: "1706.03762"\n        pages: [0]\n')),
            "whole numbers from 1 up",
        ),
        (
            _doc(_q(gold='      - paper: "1706.03762"\n        pages: ["5"]\n')),
            "whole numbers from 1 up",
        ),
        (_doc(_q(gold='      - paper: "1706.03762"\n        page: [5]\n')), "unknown fields: page"),
        (_doc(_q(qtype="unanswerable")), "must have an empty 'gold'"),
        (_doc(_q(qtype="cross_doc")), "at least 2 papers"),
        (_doc(_q(), _q()), "duplicate id 'q001' (first used on line 3)"),
        (_doc(_q(extra="    notes: [1, 2]\n")), "'notes' must be text"),
    ],
)
def test_each_mistake_is_explained(text, message):
    assert any(message in p for p in _problems(text)), _problems(text)


def test_all_problems_are_reported_together():
    text = _doc(_q(qid="q001", qtype="bad"), _q(qid="q002", answer="''"))
    problems = _problems(text)
    assert len(problems) == 2
    assert problems[0].startswith("questions.yaml:3: q001") and problems[1].startswith(
        "questions.yaml:10: q002"
    )


# --------------------------------------------------------------------------- #
# Checks against the corpus
# --------------------------------------------------------------------------- #


def test_unknown_paper_and_missing_page_are_reported():
    qs = parse(_doc(_q(qtype="multi_filter", gold=TWO_PAPERS)))
    problems = check_against_corpus(qs, {"2004.04906"}, {"2004.04906": 1})
    assert problems == [
        "line 3: q001: page 2 of 2004.04906 doesn't exist (the paper has 1 pages)",
        "line 3: q001: paper 2005.11401 is not in the corpus",
    ]
    assert check_against_corpus(qs, {"2004.04906", "2005.11401"}) == []
