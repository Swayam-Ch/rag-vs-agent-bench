"""scripts/compare_models.py and scripts/compare_retrieval.py on hand-made runs.

Both read run folders through a list file ("<name> <repeat> results/<run>"), so each test
builds a small results/ tree from the tiny fixture's rows:

    agent rows: q1 single_fact 1.0 ($0.040), q2 multi_filter 1.0 ($0.060) -> 1.00, $0.050/q
    bm25 rows:  q1 single_fact 1.0 ($0.010), q2 multi_filter 0.4 ($0.020) -> 0.70, $0.015/q
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TINY = Path(__file__).parent / "fixtures" / "results_tiny"
SCRIPTS = Path(__file__).parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(method):
    lines = (TINY / f"{method}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def make_run(root, name, method, run_rows):
    run_dir = root / "results" / name
    run_dir.mkdir(parents=True)
    info = {"config": {"methods": [method]}, "status": "finished"}
    (run_dir / "run.json").write_text(json.dumps(info), encoding="utf-8")
    text = "".join(json.dumps(r) + "\n" for r in run_rows)
    (run_dir / f"{method}.jsonl").write_text(text, encoding="utf-8")
    return f"results/{name}"


def write_list(root, filename, entries):
    path = root / "results" / filename
    path.write_text("".join(f"{n} {i} {d}\n" for n, i, d in entries), encoding="utf-8")
    return path


def with_points(run_rows, qid, points):
    out = json.loads(json.dumps(run_rows))
    for r in out:
        if r["id"] == qid:
            r["judge"]["points"] = points
    return out


def test_compare_models_reports_each_run_and_the_paired_difference(tmp_path):
    a1 = make_run(tmp_path, "a1", "agent", rows("agent"))
    a2 = make_run(tmp_path, "a2", "agent", rows("agent"))
    b1 = make_run(tmp_path, "b1", "agent", with_points(rows("agent"), "q2", 0.5))
    entries = [("model-a", 1, a1), ("model-b", 1, b1), ("model-a", 2, a2)]
    listed = write_list(tmp_path, "model-comparison-runs.txt", entries)

    compare = load("compare_models")
    text = compare.render(compare.read_list(listed))

    assert "| model-a | 2 | 1.00, 1.00 | 1.00 |" in text
    assert "| model-b | 1 | 0.75 | 0.75 |" in text
    assert "$0.050" in text  # (0.040 + 0.060) / 2, the agent's own cost
    assert "| multi_filter (1) | 1.00 | 0.50 |" in text
    assert "| model-a vs model-b | +0.25 |" in text
    assert "1 / 1 / 0" in text  # better on q2, same on q1


def test_compare_retrieval_counts_an_agent_row_cost_once(tmp_path, monkeypatch, capsys):
    bm25 = make_run(tmp_path, "r1", "bm25", rows("bm25"))
    agent = make_run(tmp_path, "m1", "agent", rows("agent"))
    retrieval = write_list(tmp_path, "retrieval-judged-runs.txt", [("bm25", 1, bm25)])
    models = write_list(tmp_path, "model-comparison-runs.txt", [("model-a", 1, agent)])

    monkeypatch.setattr(sys, "argv", ["compare_retrieval.py", str(retrieval), str(models)])
    assert load("compare_retrieval").main() == 0

    text = capsys.readouterr().out
    assert "| bm25 | 1 | 0.70 | 0.70 |" in text
    assert "$0.015 |" in text  # the answers' cost
    assert "$0.050 |" in text  # agent rows repeat the cost under "answer": not $0.100
    assert "| bm25 vs agent (model-a) | -0.30 |" in text
    saved = (tmp_path / "results" / "retrieval-judged.md").read_text(encoding="utf-8")
    assert saved + "\n" == text  # print adds one newline


def test_compare_retrieval_without_runs_says_so(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "results" / "none.txt"
    monkeypatch.setattr(sys, "argv", ["compare_retrieval.py", str(missing), str(missing)])
    assert load("compare_retrieval").main() == 1
    assert "no runs listed yet" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["compare_models", "compare_retrieval"])
def test_list_paths_resolve_from_the_repo_root_not_the_working_directory(
    tmp_path, monkeypatch, name
):
    run = make_run(tmp_path, "a1", "agent", rows("agent"))
    listed = write_list(tmp_path, "x-runs.txt", [("model-a", 1, run)])
    monkeypatch.chdir(tmp_path / "results")  # somewhere the relative path does not exist
    runs = load(name).read_list(listed)
    assert all(d.exists() for dirs in runs.values() for d in dirs)
