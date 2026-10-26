"""`ragbench summarize`: markdown tables from a results folder.

The tiny fixture (tests/fixtures/results_tiny), worked by hand:

    bm25:  q1 recall@5 1.0, RR 0.5,  paper 1.0, points 1.0, answer $0.010 in 3 s
           q2 recall@5 0.5, RR 0.25, paper 0.5, points 0.4, answer $0.020 in 5 s
           -> recall@5 0.75, MRR 0.375, paper 0.75, accuracy 0.70,
              cost (0.010 + 0.020) / 2 = $0.015, time (0+3 + 0+5) / 2 = 4.0 s
    agent: q1 recall@5 0.0, RR 0.2, points 1.0, $0.040, 6 s (answering is inside its search)
           q2 recall@5 1.0, RR 1.0, points 1.0, $0.060, 10 s
           -> recall@5 0.50, MRR 0.60, paper 1.00, accuracy 1.00, $0.050, 8.0 s
    judging: 0.004 + 0.006 + 0.005 + 0.005 = $0.020
"""

from pathlib import Path

import pytest

from ragbench import cli
from ragbench.eval.summarize import render

TINY = Path(__file__).parent / "fixtures" / "results_tiny"


def test_main_table_matches_the_hand_calculation():
    text = render(TINY)
    assert "| **bm25** | 0.75 | 0.38 | 0.75 | 0.70 | $0.015 | 4.0 s |" in text
    assert "| **agent** | 0.50 | 0.60 | 1.00 | 1.00 | $0.050 | 8.0 s |" in text


def test_accuracy_by_type_and_author():
    text = render(TINY)
    assert "| single_fact (1) | 1.00 | 1.00 |" in text
    assert "| multi_filter (1) | 0.40 | 1.00 |" in text
    assert "| question type | bm25 | agent |" in text
    assert "| swayam (1) | 1.00 | 1.00 |" in text and "| claude (1) | 0.40 | 1.00 |" in text


def test_run_details_and_problems():
    text = render(TINY)
    assert "status **finished**, commit `0123456`, config `abc123def456`" in text
    assert "judged by `claude-opus-5-5` (judge-v2), judging cost $0.020" in text
    assert "- problems: bm25: check q2 by hand" in text
    assert "agent: `claude-sonnet-5-5`, at most 15 turns" in text


def test_numbers_come_from_the_rows_not_run_json():
    # the fixture's run.json "summary" is junk on purpose; the table is still right
    assert "ignored" not in render(TINY)


def test_tables_are_valid_markdown():
    for line in render(TINY).splitlines():
        if line.startswith("|"):
            assert line.endswith("|")
    header = [line for line in render(TINY).splitlines() if line.startswith("| method")][0]
    separator_index = render(TINY).splitlines().index(header) + 1
    assert render(TINY).splitlines()[separator_index].startswith("|:---|---:|")


def test_retrieval_only_run_has_no_accuracy_tables(tmp_path, processed, capsys):
    qfile = tmp_path / "q.yaml"
    qfile.write_text(
        "version: 1\nquestions:\n  - id: q1\n    type: single_fact\n    question: heads?\n"
        '    answer: Eight heads.\n    gold:\n      - paper: "1706.03762"\n        pages: [1]\n'
    )
    out = tmp_path / "results"
    argv = ["run", "--methods", "bm25", "--processed", str(processed)]
    assert cli.main([*argv, "--question-file", str(qfile), "--out", str(out)]) == 0
    capsys.readouterr()
    [run_dir] = out.iterdir()
    assert cli.main(["summarize", str(run_dir)]) == 0
    text = capsys.readouterr().out
    assert "| **bm25** |" in text and "| – |" in text  # no accuracy without --judge
    assert "Accuracy by question type" not in text
    assert "- problems: none" in text


def test_missing_folder_is_a_clear_error(tmp_path, capsys):
    (tmp_path / "run.json").write_text('{"config": {"methods": ["bm25"]}}')
    assert cli.main(["summarize", str(tmp_path)]) == 1
    assert "no <method>.jsonl files" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):
        render(tmp_path / "nope")
