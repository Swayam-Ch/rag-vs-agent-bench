"""`ragbench sweep`: retrieval-only comparison of setups, with stand-in models."""

import argparse
import json

import pytest
from fakes import HashEmbedder

from ragbench import cli
from ragbench.eval import sweep
from ragbench.eval.sweep import CONFIGS, Builder, Config, main_sweep

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
    type: negation
    question: Which papers do not mention zebras?
    answer: Both.
    gold:
      - paper: "1706.03762"
        pages: []
"""


class OverlapScorer:
    name = "overlap-scorer"

    def __init__(self):
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        q = set(query.lower().split())
        return [len(q & set(t.lower().split())) - 1.0 for t in texts]


@pytest.fixture
def setup(processed, tmp_path):
    qfile = tmp_path / "q.yaml"
    qfile.write_text(QUESTIONS)
    made = []

    def make_embedder(spec):
        made.append(spec)
        return HashEmbedder(name=spec["model_name"])

    scorer = OverlapScorer()
    builder = Builder(processed, make_embedder=make_embedder, make_scorer=lambda: scorer)
    args = argparse.Namespace(
        configs=None, question_file=qfile, processed=processed, out=tmp_path / "results"
    )
    return args, builder, made, scorer


def _folder(args):
    [folder] = sorted(args.out.iterdir())
    return folder, json.loads((folder / "run.json").read_text())


def test_every_configuration_runs_and_shares_models(setup, capsys):
    args, builder, made, scorer = setup
    assert main_sweep(args, builder) == 0
    folder, info = _folder(args)
    assert folder.name.startswith("sweep-")
    assert sorted(p.stem for p in folder.glob("*.jsonl")) == sorted(CONFIGS)
    # each embedding model is loaded once, however many configurations use it
    assert [m["model_name"] for m in made] == [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
    ]
    assert made[1]["query_prefix"].startswith("Represent this sentence")
    assert scorer.calls > 0
    out = capsys.readouterr().out
    assert "| hybrid-bge+rerank | paragraph |" in out and "| bm25-fixed | fixed |" in out
    assert (folder / "summary.md").read_text() in out


def test_run_json_records_each_configuration(setup):
    args, builder, _, _ = setup
    args.configs = "bm25,hybrid-bge+rerank"
    assert main_sweep(args, builder) == 0
    _, info = _folder(args)
    assert info["config"]["k"] == 20
    assert info["config"]["sweep"]["hybrid-bge+rerank"] == {
        "method": "hybrid",
        "chunks": "paragraph",
        "embedder": "bge",
        "rerank": True,
    }
    rerank = info["config"]["searchers"]["hybrid-bge+rerank"]
    assert rerank["class"] == "RerankSearcher" and rerank["reranker"] == "overlap-scorer"
    assert rerank["reranks"]["class"] == "HybridSearcher"
    assert rerank["reranks"]["fuses"][1]["embedder"] == "BAAI/bge-base-en-v1.5"
    assert info["config"]["searchers"]["bm25"]["class"] == "BM25Searcher"  # not the wrapper


def test_rows_have_recall_at_20_and_skip_questions_without_evidence(setup):
    args, builder, _, _ = setup
    args.configs = "bm25"
    assert main_sweep(args, builder) == 0
    folder, _ = _folder(args)
    q1, q2 = [json.loads(line) for line in (folder / "bm25.jsonl").read_text().splitlines()]
    assert q1["method"] == "bm25" and "recall@20" in q1["metrics"]
    assert q2["metrics"] is None  # negation: nothing to retrieve


def test_unknown_configuration_is_refused(setup, capsys):
    args, builder, _, _ = setup
    args.configs = "bm25,magic"
    assert main_sweep(args, builder) == 1
    assert "unknown configuration(s) magic" in capsys.readouterr().err
    assert not args.out.exists()


def test_a_model_that_fails_to_load_is_a_clear_error(setup, capsys):
    args, _, _, _ = setup

    def offline(spec):
        raise RuntimeError("could not look up the current revision of BAAI/bge-base-en-v1.5")

    args.configs = "vector-bge"
    assert main_sweep(args, Builder(args.processed, make_embedder=offline)) == 1
    assert "could not look up" in capsys.readouterr().err


def test_table_formats_missing_values():
    summary = {"x": {"metrics": {"recall@5": 0.5}, "seconds": 2.0, "questions": 4}}
    [_, _, row] = sweep.table(summary, {"x": Config("bm25")})
    assert row == "| x | paragraph | 0.50 | – | – | – | – | 0.50 s |"


def test_cli_is_wired(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(sweep, "main_sweep", lambda args: seen.setdefault("args", args) and 0)
    cli.main(["sweep", "--configs", "bm25", "--out", str(tmp_path)])
    assert seen["args"].configs == "bm25"
