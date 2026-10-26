"""`ragbench chart`: the accuracy-by-type PNG for the README."""

import json
import shutil
from pathlib import Path

import pytest

from ragbench import cli
from ragbench.eval.chart import METHOD_COLORS, accuracy_by_type, plot

TINY = Path(__file__).parent / "fixtures" / "results_tiny"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_data_matches_the_fixture():
    # bm25: q1 single_fact 1.0, q2 multi_filter 0.4; agent: 1.0 and 1.0
    data = accuracy_by_type(TINY)
    assert data["types"] == [("single_fact", 1), ("multi_filter", 1)]  # format order
    assert data["methods"] == {"bm25": [1.0, 0.4], "agent": [1.0, 1.0]}


def test_plot_writes_a_png(tmp_path):
    out = plot(TINY, tmp_path / "chart.png")
    assert out.read_bytes()[:8] == PNG_MAGIC and out.stat().st_size > 10_000


def test_cli_saves_next_to_the_results(tmp_path, capsys):
    run = tmp_path / "run"
    shutil.copytree(TINY, run)
    assert cli.main(["chart", str(run)]) == 0
    assert capsys.readouterr().out.strip() == str(run / "accuracy_by_type.png")
    assert (run / "accuracy_by_type.png").read_bytes()[:8] == PNG_MAGIC


def test_a_run_without_judged_answers_is_refused(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"config": {"methods": ["bm25"]}}))
    row = {"id": "q1", "type": "single_fact", "seconds": 0.0, "error": "", "metrics": None}
    (run / "bm25.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="run it with --judge"):
        accuracy_by_type(run)
    assert cli.main(["chart", str(run)]) == 1
    assert "--judge" in capsys.readouterr().err


def test_every_method_has_a_fixed_distinct_colour():
    assert set(METHOD_COLORS) == set(cli.METHODS)
    assert len(set(METHOD_COLORS.values())) == len(METHOD_COLORS)
