"""The benchmark runner: `ragbench run` writes a self-describing results folder."""

import json

import pytest
from fakes import GradingClient, HashEmbedder, PolicyClient

from ragbench import cli
from ragbench.eval import grade
from ragbench.eval import run as run_mod
from ragbench.eval.questions import parse
from ragbench.search import agent_search, vector_search
from ragbench.search.base import SearchResult

QUESTIONS = """\
version: 1
questions:
  - id: q1
    type: single_fact
    question: How does the number of attention heads change BLEU?
    answer: See Table 3.
    gold:
      - paper: "1706.03762"
        pages: [1]
  - id: q2
    type: multi_filter
    question: Which papers compare against BM25 on Natural Questions?
    answer: DPR.
    gold:
      - paper: "2004.04906"
        pages: [1]
  - id: q3
    type: unanswerable
    question: What did the zebra cost?
    answer: Not reported.
    gold: []
"""


@pytest.fixture
def setup(processed, tmp_path):
    qfile = tmp_path / "questions.yaml"
    qfile.write_text(QUESTIONS)
    out = tmp_path / "results"
    base = ["--processed", str(processed), "--question-file", str(qfile), "--out", str(out)]
    return base, out


def _run(base, *argv):
    return cli.main(["run", *base, *argv])


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _only_run(out):
    [folder] = sorted(out.iterdir())
    return folder, json.loads((folder / "run.json").read_text())


# --------------------------------------------------------------------------- #
# The folder: one jsonl per method, one row per question, and run.json
# --------------------------------------------------------------------------- #


def test_run_writes_a_self_describing_folder(setup, capsys):
    base, out = setup
    assert _run(base, "--methods", "keyword,bm25") == 0
    folder, info = _only_run(out)
    assert sorted(p.name for p in folder.iterdir()) == ["bm25.jsonl", "keyword.jsonl", "run.json"]

    assert info["status"] == "finished" and info["started"] and info["finished"]
    assert info["config"]["methods"] == ["keyword", "bm25"] and info["config"]["k"] == 10
    bm25 = info["config"]["searchers"]["bm25"]
    assert (bm25["class"], bm25["k1"], bm25["b"], bm25["idf_variant"]) == (
        "BM25Searcher",
        1.5,
        0.75,
        "lucene",
    )
    assert info["inputs"]["questions"]["ids"] == ["q1", "q2", "q3"]
    assert len(info["inputs"]["questions"]["sha256"]) == 64
    assert info["inputs"]["chunks"]["file"] == cli.DEFAULT_CHUNKS
    assert info["packages"]["numpy"] and "git" in info and info["python"]

    rows = _rows(folder / "bm25.jsonl")
    assert [r["id"] for r in rows] == ["q1", "q2", "q3"]
    q1, _, q3 = rows
    assert q1["question"].startswith("How does") and q1["method"] == "bm25"
    assert q1["metrics"]["recall@1"] == 1.0 and q1["metrics"]["rr"] == 1.0
    assert set(q1["metrics"]) == {
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "recall@all",
        "paper_recall@all",
        "rr",
        "paper_rr",
    }
    top = q1["results"][0]
    assert (top["rank"], top["paper"], top["page"]) == (1, "1706.03762", 1)
    assert top["chunk_id"] and len(top["preview"]) <= run_mod.PREVIEW_CHARS
    assert q3["metrics"] is None  # unanswerable: nothing to retrieve, not a 0

    summary = info["summary"]["bm25"]
    assert (summary["questions"], summary["scored"], summary["errors"]) == (3, 2, 0)
    assert summary["metrics"]["recall@1"] == 1.0  # both answerable questions found at rank 1

    printed = capsys.readouterr().out
    assert "recall@1" in printed and "bm25" in printed and f"saved to {folder}" in printed


def test_same_config_gives_a_new_folder_with_the_same_hash(setup):
    base, out = setup
    assert _run(base, "--methods", "bm25") == 0
    assert _run(base, "--methods", "bm25") == 0
    assert _run(base, "--methods", "bm25", "-k", "5") == 0
    folders = sorted(out.iterdir())
    assert len(folders) == 3  # never overwrites, even within the same second
    hashes = [json.loads((f / "run.json").read_text())["config_hash"] for f in folders]
    assert hashes[0] == hashes[1] != hashes[2]


def test_filters_pick_methods_and_questions(setup):
    base, out = setup
    assert _run(base, "--methods", "keyword", "--questions", "q3,q1") == 0
    folder, info = _only_run(out)
    assert [p.name for p in folder.glob("*.jsonl")] == ["keyword.jsonl"]
    assert [r["id"] for r in _rows(folder / "keyword.jsonl")] == ["q1", "q3"]  # file order
    assert info["inputs"]["questions"]["ids"] == ["q1", "q3"]


def test_k_limits_results_and_metrics(setup):
    base, out = setup
    assert _run(base, "--methods", "bm25", "-k", "3") == 0
    folder, _ = _only_run(out)
    q1 = _rows(folder / "bm25.jsonl")[0]
    assert len(q1["results"]) <= 3
    assert "recall@5" not in q1["metrics"] and "recall@3" in q1["metrics"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--methods", "bm25,magic"], "unknown method(s) magic"),
        (["--questions", "q1,q99"], "unknown question id(s): q99"),
        (["--question-file", "nope.yaml"], "nope.yaml"),
    ],
)
def test_bad_arguments_fail_before_creating_a_folder(setup, capsys, argv, message):
    base, out = setup
    assert _run(base, *argv) == 1
    assert message in capsys.readouterr().err
    assert not out.exists() or not any(out.iterdir())


def test_invalid_question_file_lists_its_problems(setup, tmp_path, capsys):
    base, _ = setup
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nquestions:\n  - id: q1\n    type: magic\n")
    assert _run(base, "--question-file", str(bad)) == 1
    assert "unknown type 'magic'" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Vector, hybrid and the agent
# --------------------------------------------------------------------------- #


def test_vector_and_hybrid_share_one_embedding_model(setup, monkeypatch):
    built = []

    class CountingEmbedder(HashEmbedder):
        def __init__(self):
            super().__init__()
            built.append(self)

    monkeypatch.setattr(vector_search, "SentenceTransformerEmbedder", CountingEmbedder)
    base, out = setup
    assert _run(base, "--methods", "vector,hybrid") == 0
    assert len(built) == 1
    _, info = _only_run(out)
    hybrid = info["config"]["searchers"]["hybrid"]
    assert [s["class"] for s in hybrid["fuses"]] == ["BM25Searcher", "VectorSearcher"]
    assert hybrid["fuses"][1]["embedder"] == built[0].name  # the model is recorded
    assert hybrid["rrf_k"] == 60


def test_agent_rows_have_answer_tokens_cost_and_all_evidence(setup, monkeypatch):
    monkeypatch.setattr(agent_search, "_default_client", PolicyClient)
    base, out = setup
    assert _run(base, "--methods", "agent", "-k", "1", "--questions", "q1") == 0
    folder, info = _only_run(out)
    [row] = _rows(folder / "agent.jsonl")
    agent = row["agent"]
    assert agent["answer"].startswith("Done.") and agent["stop"] == "answered"
    assert agent["input_tokens"] > 0 and agent["cost_usd"] > 0
    assert [c["name"] for c in agent["tool_calls"]] == ["grep", "read_file"]
    assert "messages" not in agent and "evidence" not in agent
    # the agent's evidence has no natural top k: -k 1 must not throw most of it away
    assert len(row["results"]) > 1 and "recall@all" in row["metrics"]
    assert info["config"]["searchers"]["agent"]["model"] == "claude-sonnet-5-5"
    assert info["summary"]["agent"]["agent"]["stops"] == {"answered": 1}
    assert info["summary"]["agent"]["agent"]["cost_usd"] == pytest.approx(agent["cost_usd"])
    assert list((folder / "transcripts").glob("*.json"))  # kept locally, git-ignored


# --------------------------------------------------------------------------- #
# Failures: recorded, never fatal; Ctrl-C leaves a readable partial run
# --------------------------------------------------------------------------- #


class Broken:
    name = "broken"

    def __init__(self, exc):
        self.exc = exc

    def search(self, query, k=10):
        raise self.exc


QS = parse(QUESTIONS).questions


def test_a_failing_search_is_recorded_and_scores_zero():
    row = run_mod.run_one(Broken(RuntimeError("index missing")), QS[0], 10, {})
    assert row["error"] == "RuntimeError: index missing"
    assert row["results"] == [] and row["metrics"]["recall@10"] == 0.0


def test_errors_are_counted_and_the_run_finishes(tmp_path):
    info = run_mod.run_benchmark(
        QS,
        {"broken": Broken(RuntimeError("boom"))},
        k=10,
        arxiv_of={},
        out_dir=tmp_path,
        info={},
    )
    assert info["status"] == "finished" and info["summary"]["broken"]["errors"] == 3
    assert len(_rows(tmp_path / "broken.jsonl")) == 3


def test_ctrl_c_marks_the_run_interrupted(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run_mod.run_benchmark(
            QS,
            {"broken": Broken(KeyboardInterrupt())},
            k=10,
            arxiv_of={},
            out_dir=tmp_path,
            info={"config": {"k": 10}},
        )
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["status"] == "interrupted" and info["finished"] and info["config"] == {"k": 10}


# --------------------------------------------------------------------------- #
# Small pieces
# --------------------------------------------------------------------------- #


def test_run_dirs_never_collide(tmp_path):
    from datetime import UTC, datetime

    now = datetime(2026, 10, 26, 10, 15, 0, tzinfo=UTC)
    names = [run_mod.new_run_dir(tmp_path, now).name for _ in range(3)]
    assert names == ["2026-10-26T101500Z", "2026-10-26T101500Z-2", "2026-10-26T101500Z-3"]


def test_git_info_outside_a_repository_is_none(tmp_path):
    assert run_mod.git_info(tmp_path) is None


def test_config_hash_ignores_key_order():
    assert run_mod.config_hash({"a": 1, "b": [1, 2]}) == run_mod.config_hash({"b": [1, 2], "a": 1})
    assert run_mod.config_hash({"a": 1}) != run_mod.config_hash({"a": 2})


def test_summary_table_shows_dashes_for_missing_metrics():
    summary = {
        "agent": {
            "metrics": {"recall@1": None, "rr": 0.5},
            "errors": 1,
            "seconds": 2.0,
            "agent": {"cost_usd": 0.0123},
        }
    }
    [head, line] = run_mod.format_summary(summary, k=1)
    assert "recall@1" in head and "recall@3" not in head
    assert line.split()[1] == "-" and "0.500" in line and "cost≈$0.012" in line


def test_results_keep_their_page_range():
    r = SearchResult("abc", 0, 5, 4, 5, 1.0, "text")
    row = run_mod._result_row(1, r, "2004.04906")
    assert (row["page"], row["page_end"], row["paper"]) == (4, 5, "2004.04906")


# --------------------------------------------------------------------------- #
# --judge: answers and verdicts (#18)
# --------------------------------------------------------------------------- #


@pytest.fixture
def grading(monkeypatch):
    client = GradingClient()
    monkeypatch.setattr(grade, "default_client", lambda: client)
    return client


def test_judge_adds_answer_verdict_tokens_and_cost_to_every_row(setup, grading, capsys):
    base, out = setup
    assert _run(base, "--methods", "bm25", "--judge") == 0
    folder, info = _only_run(out)
    rows = _rows(folder / "bm25.jsonl")
    assert len(grading.answer_calls()) == len(grading.judge_calls()) == 3  # every question
    for row in rows:
        answer, judge = row["answer"], row["judge"]
        assert answer["source"] == "generated" and answer["text"].startswith("Eight heads")
        assert answer["input_tokens"] == 1000 and answer["cost_usd"] > 0 and answer["seconds"] >= 0
        assert (judge["verdict"], judge["points"]) == ("correct", 1.0)
        assert judge["model"] == "claude-opus-5-5" and judge["cost_usd"] > 0
    # the answer is written from the passages this method retrieved
    first_prompt = grading.answer_calls()[0]["messages"][0]["content"]
    assert rows[0]["results"][0]["preview"][:30] in " ".join(first_prompt.split())

    answers = info["summary"]["bm25"]["answers"]
    assert answers["accuracy"] == 1.0 and answers["correct"] == 3 and answers["unjudged"] == 0
    assert set(answers["by_type"]) == {"single_fact", "multi_filter", "unanswerable"}
    assert answers["answer_cost_usd"] == pytest.approx(3 * (1000 * 2 + 100 * 10) / 1e6)
    assert answers["seconds_to_answer"] == pytest.approx(
        sum(r["seconds"] + r["answer"]["seconds"] for r in rows), abs=1e-3
    )
    grader = info["config"]["grader"]
    assert (grader["answer_model"], grader["judge_model"]) == (
        "claude-sonnet-5-5",
        "claude-opus-5-5",
    )
    printed = capsys.readouterr()
    assert "accuracy" in printed.out and "judging cost $" in printed.out
    assert "bm25 q1" in printed.err and "correct" in printed.err


def test_judging_changes_the_config_hash(setup, grading):
    base, out = setup
    assert _run(base, "--methods", "bm25") == 0
    assert _run(base, "--methods", "bm25", "--judge") == 0
    assert _run(base, "--methods", "bm25", "--judge", "--judge-model", "claude-sonnet-5-5") == 0
    hashes = [
        json.loads((f / "run.json").read_text())["config_hash"] for f in sorted(out.iterdir())
    ]
    assert len(set(hashes)) == 3


def test_without_judge_nothing_is_answered_or_paid_for(setup, grading):
    base, out = setup
    assert _run(base, "--methods", "bm25") == 0
    folder, info = _only_run(out)
    assert grading.calls == [] and info["config"]["grader"] is None
    assert "answer" not in _rows(folder / "bm25.jsonl")[0]


def test_the_agent_is_judged_on_its_own_answer(setup, grading, monkeypatch):
    monkeypatch.setattr(agent_search, "_default_client", PolicyClient)
    base, out = setup
    assert _run(base, "--methods", "agent", "--judge", "--questions", "q1") == 0
    folder, info = _only_run(out)
    [row] = _rows(folder / "agent.jsonl")
    assert grading.answer_calls() == []  # no answer written for it
    assert row["answer"]["source"] == "agent" and row["answer"]["text"].startswith("Done.")
    assert row["answer"]["cost_usd"] == row["agent"]["cost_usd"]
    judged = grading.judge_calls()[0]["messages"][0]["content"]
    assert "<answer>\nDone." in judged
    assert info["summary"]["agent"]["answers"]["answer_cost_usd"] == pytest.approx(
        row["agent"]["cost_usd"]
    )


def test_a_failed_answer_call_is_unjudged_not_wrong(setup, monkeypatch):
    client = GradingClient(fail="answer")
    monkeypatch.setattr(grade, "default_client", lambda: client)
    base, out = setup
    assert _run(base, "--methods", "bm25", "--judge", "--questions", "q1") == 0
    folder, info = _only_run(out)
    [row] = _rows(folder / "bm25.jsonl")
    assert row["answer"]["error"] == "RuntimeError: answer API down"
    assert row["judge"]["verdict"] is None and client.judge_calls() == []
    answers = info["summary"]["bm25"]["answers"]
    assert answers["unjudged"] == 1 and answers["accuracy"] is None  # not a 0


def test_a_failed_search_counts_as_a_wrong_answer():
    grader = grade.Grader(GradingClient(), {})
    row = run_mod.run_one(Broken(RuntimeError("index missing")), QS[0], 10, {}, grader)
    assert row["answer"] == {
        "source": "none",
        "text": "",
        "cost_usd": 0.0,
        "error": "RuntimeError: index missing",
    }
    assert (row["judge"]["verdict"], row["judge"]["reason"]) == ("wrong", "no answer was given")
    assert grader.client.calls == []


def test_unknown_prices_make_the_total_unknown_not_smaller():
    assert run_mod._total([0.1, None, 0.2]) is None
    assert run_mod._total([0.1, 0.2]) == pytest.approx(0.3)
