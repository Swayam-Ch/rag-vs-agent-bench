"""`ragbench stats`: bootstrap intervals and the agent citation check."""

import json
import shutil
from pathlib import Path

import pytest

from ragbench import cli
from ragbench.eval.stats import bootstrap_mean, check_citations, paired, render

TINY = Path(__file__).parent / "fixtures" / "results_tiny"


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def test_constant_values_have_a_zero_width_interval():
    assert bootstrap_mean([1.0] * 20) == (1.0, 1.0, 1.0)


def test_interval_contains_the_mean_and_is_reproducible():
    values = [1, 0, 1, 1, 0.5, 0, 1, 1, 0, 1]  # mean 0.65
    mean, low, high = bootstrap_mean(values)
    assert mean == pytest.approx(0.65) and low < mean < high
    assert 0 <= low and high <= 1
    assert bootstrap_mean(values) == (mean, low, high)  # fixed seed


def test_more_questions_give_a_narrower_interval():
    _, low10, high10 = bootstrap_mean([1, 0] * 5)
    _, low100, high100 = bootstrap_mean([1, 0] * 50)
    assert high100 - low100 < high10 - low10


def test_empty_is_undefined():
    assert bootstrap_mean([]) == (None, None, None)


def test_paired_difference_counts_and_uses_shared_questions_only():
    a = {"q1": 1.0, "q2": 1.0, "q3": 0.5, "q4": 0.0}
    b = {"q1": 0.0, "q2": 1.0, "q3": 1.0}  # q4 unjudged for b: left out
    p = paired(a, b)
    assert (p["n"], p["wins"], p["ties"], p["losses"]) == (3, 1, 1, 1)
    assert p["diff"] == pytest.approx((1 + 0 - 0.5) / 3)


def test_a_consistent_winner_has_an_interval_above_zero():
    a = {f"q{i}": 1.0 for i in range(20)}
    b = {f"q{i}": 0.0 if i < 12 else 1.0 for i in range(20)}  # a better on 12 of 20
    p = paired(a, b)
    assert p["low"] > 0 and p["diff"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #


def _row(citations, shown):
    return {
        "agent": {"citations": citations},
        "results": [{"paper": p, "page": a, "page_end": b} for p, a, b in shown],
    }


def test_citations_are_checked_against_gold_and_against_what_was_shown():
    gold = [{"paper": "2004.04906", "pages": [5]}, {"paper": "2112.09118", "pages": [7]}]
    row = _row(
        [
            {"ref": "2004.04906", "doc_id": "d1", "page": 5},  # gold, shown (in 4-5)
            {"ref": "2004.04906v2", "doc_id": "d1", "page": 9},  # right paper, wrong page
            {"ref": "c0ffee123456", "doc_id": "c0ffee123456", "page": 7},  # doc ID -> arXiv
        ],
        shown=[("2004.04906", 4, 5), ("2004.04906", 9, 9)],
    )
    out = check_citations(row, gold, {"c0ffee123456": "2112.09118"})
    assert [(c["paper"], c["page"], c["gold"], c["seen"]) for c in out["citations"]] == [
        ("2004.04906", 5, True, True),
        ("2004.04906", 9, False, True),
        ("2112.09118", 7, True, False),  # cited a page it was never shown
    ]
    assert (out["gold_entries"], out["gold_cited"]) == (2, 2)


def test_a_whole_paper_gold_entry_accepts_any_page():
    out = check_citations(
        _row([{"ref": "2208.03299", "doc_id": "a", "page": 3}], shown=[]),
        [{"paper": "2208.03299", "pages": []}],
        {},
    )
    assert out["citations"][0]["gold"] is True and out["gold_cited"] == 1


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

QUESTIONS = """\
version: 1
questions:
  - id: q1
    type: single_fact
    question: How many heads?
    answer: Eight heads.
    gold:
      - paper: "1706.03762"
        pages: [5]
  - id: q2
    type: multi_filter
    question: Which papers use FAISS?
    answer: DPR.
    gold:
      - paper: "2004.04906"
        pages: [3]
"""


@pytest.fixture
def run(tmp_path):
    folder = tmp_path / "run"
    shutil.copytree(TINY, folder)
    rows = [json.loads(line) for line in (folder / "agent.jsonl").read_text().splitlines()]
    rows[0]["agent"]["citations"] = [{"ref": "1706.03762", "doc_id": "a", "page": 5}]
    rows[0]["results"] = [{"paper": "1706.03762", "doc_id": "a", "page": 5, "page_end": 5}]
    rows[1]["agent"]["citations"] = [{"ref": "2004.04906", "doc_id": "b", "page": 8}]
    (folder / "agent.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    qfile = tmp_path / "q.yaml"
    qfile.write_text(QUESTIONS)
    return folder, qfile


def test_report_has_intervals_differences_and_citations(run):
    folder, qfile = run
    text = render(folder, qfile)
    assert "| agent | 1.00 | 1.00 to 1.00 | 2 |" in text
    assert "| agent vs bm25 | +0.30 |" in text  # (0 + 0.6) / 2; the agent always first
    assert "citations on a gold page: 1 of 2 (50%)" in text
    assert "gold evidence entries the answers cited: 1 of 2 (50%)" in text
    assert "citations of a page the agent had been shown: 1 of 2 (50%)" in text
    assert (
        "never shown, so cited from memory: q2 (2004.04906 p.8) (0 of them on a gold page)" in text
    )


def test_cli_writes_stats_md(run, capsys):
    folder, qfile = run
    assert cli.main(["stats", str(folder), "--question-file", str(qfile)]) == 0
    assert (folder / "stats.md").read_text() == capsys.readouterr().out


def test_a_run_without_judged_answers_is_refused(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps({"config": {"methods": ["bm25"]}}))
    (tmp_path / "bm25.jsonl").write_text(json.dumps({"id": "q1", "metrics": None}) + "\n")
    assert cli.main(["stats", str(tmp_path)]) == 1
    assert "--judge" in capsys.readouterr().err
