import json
from pathlib import Path

import pytest
from fakes import HashEmbedder, PolicyClient

from ragbench import cli
from ragbench.pipeline.chunk import main as chunk_main
from ragbench.search import agent_search, vector_search
from ragbench.search.agent_search import estimate_cost


def _run(capsys, *argv):
    code = cli.main(["search", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------- #
# Chunk-based methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["keyword", "bm25"])
def test_text_output_shows_rank_paper_page_and_snippet(processed, capsys, method):
    code, out, _ = _run(capsys, "--processed", str(processed), "-m", method, "attention heads")
    assert code == 0
    first = out.splitlines()[1]
    assert first.lstrip().startswith("1.")
    assert "1706.03762" in first and "p.1" in first and "Attention Is All You Need" in first
    assert "number of attention heads" in out


def test_k_limits_the_number_of_results(processed, capsys):
    _, out, _ = _run(capsys, "--processed", str(processed), "-k", "1", "retrieval attention")
    assert sum(line.lstrip().startswith(("1.", "2.")) for line in out.splitlines()) == 1


def test_json_output_is_valid_and_complete(processed, capsys):
    code, out, _ = _run(capsys, "--processed", str(processed), "--json", "BM25")
    payload = json.loads(out)
    assert code == 0 and payload["method"] == "bm25" and payload["query"] == "BM25"
    [result] = payload["results"]
    assert "BM25" in result["text"] and result["chunk_id"]


@pytest.mark.parametrize("method", ["vector", "hybrid"])
def test_vector_and_hybrid_build_from_the_cli(processed, capsys, monkeypatch, method):
    monkeypatch.setattr(vector_search, "SentenceTransformerEmbedder", HashEmbedder)
    code, out, _ = _run(capsys, "--processed", str(processed), "-m", method, "attention heads")
    assert code == 0 and f"results ({method})" in out
    assert list((processed / "cache").glob("embeddings.*.npy"))  # cached for next time


def test_no_results(processed, capsys):
    code, out, _ = _run(capsys, "--processed", str(processed), "zebra")
    assert code == 0 and out.strip() == "no results"


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


def test_agent_prints_answer_stats_and_evidence(processed, capsys, monkeypatch):
    monkeypatch.setattr(agent_search, "_default_client", PolicyClient)
    code, out, _ = _run(capsys, "--processed", str(processed), "-m", "agent", "attention heads")
    assert code == 0
    assert out.startswith("Done.")
    assert "stop=answered turns=3" in out and "cost≈$" in out
    assert "tools (2):" in out
    assert '    grep {"pattern": "attention|heads"}' in out  # one call per line, args visible
    assert "evidence (agent):" in out
    assert list((processed / "transcripts").glob("*.json"))


def test_agent_json_includes_run_details_but_not_the_transcript(processed, capsys, monkeypatch):
    monkeypatch.setattr(agent_search, "_default_client", PolicyClient)
    _, out, _ = _run(capsys, "--processed", str(processed), "-m", "agent", "--json", "BM25")
    payload = json.loads(out)
    assert payload["agent"]["stop"] == "answered"
    assert "messages" not in payload["agent"]
    assert payload["results"]


# --------------------------------------------------------------------------- #
# Choosing the chunks file
# --------------------------------------------------------------------------- #


def test_single_non_default_chunks_file_is_used(processed, capsys):
    (processed / cli.DEFAULT_CHUNKS).rename(processed / "chunks.paragraph.500.jsonl")
    code, _, err = _run(capsys, "--processed", str(processed), "attention")
    assert code == 0 and "using chunks.paragraph.500.jsonl" in err


def test_several_chunks_files_need_a_choice(processed, capsys):
    (processed / cli.DEFAULT_CHUNKS).rename(processed / "chunks.paragraph.500.jsonl")
    assert chunk_main([str(processed), "--method", "fixed"]) == 0
    code, _, err = _run(capsys, "--processed", str(processed), "attention")
    assert code == 1 and "pick one with --chunks" in err
    code, _, _ = _run(
        capsys,
        "--processed",
        str(processed),
        "--chunks",
        str(processed / "chunks.fixed.1000-200.jsonl"),
        "attention",
    )
    assert code == 0


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_missing_corpus(tmp_path, capsys):
    code, _, err = _run(capsys, "--processed", str(tmp_path), "x")
    assert code == 1 and "ragbench-ingest" in err


def test_missing_chunks(processed, capsys):
    (processed / cli.DEFAULT_CHUNKS).unlink()
    code, _, err = _run(capsys, "--processed", str(processed), "x")
    assert code == 1 and "ragbench-chunk" in err


def test_empty_query(processed, capsys):
    code, _, err = _run(capsys, "--processed", str(processed), "   ")
    assert code == 1 and "empty" in err


@pytest.mark.parametrize("argv", [["search", "-k", "0", "q"], ["search", "-m", "magic", "q"], []])
def test_bad_arguments_exit_with_usage(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# Cost estimate
# --------------------------------------------------------------------------- #


def test_estimate_cost():
    # the real run on the attention-heads question: 9,789 in, 562 out, Sonnet 5.5
    assert estimate_cost("claude-sonnet-5-5", 9_789, 562) == pytest.approx(0.025198)
    assert estimate_cost("some-future-model", 1000, 1000) is None


# --------------------------------------------------------------------------- #
# ragbench questions
# --------------------------------------------------------------------------- #

GOOD = """version: 1
questions:
  - id: q001
    type: single_fact
    question: How many heads?
    answer: Eight.
    gold:
      - paper: "1706.03762"
        pages: [5]
"""


def _write(tmp_path, text):
    path = tmp_path / "questions.yaml"
    path.write_text(text)
    return path


def _metadata(tmp_path, ids=("1706.03762",)):
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "arxiv_id": i,
                    "version": "v1",
                    "title": "T",
                    "authors": [],
                    "published": "2017-01-01",
                    "category": "cs.CL",
                }
            )
            + "\n"
            for i in ids
        )
    )
    return path


def test_questions_command_reports_ok_and_missing_types(tmp_path, capsys):
    code = cli.main(
        [
            "questions",
            str(_write(tmp_path, GOOD)),
            "--metadata",
            str(_metadata(tmp_path)),
            "--processed",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "1 questions OK (1 single_fact)" in out
    assert "no questions yet of type: multi_filter" in out


def test_questions_command_lists_problems(tmp_path, capsys):
    bad = GOOD.replace("single_fact", "single-fact")
    code = cli.main(
        [
            "questions",
            str(_write(tmp_path, bad)),
            "--metadata",
            str(_metadata(tmp_path)),
            "--processed",
            str(tmp_path),
        ]
    )
    err = capsys.readouterr().err
    assert code == 1 and "questions.yaml:3: q001: unknown type 'single-fact'" in err


def test_questions_command_checks_papers_and_pages(tmp_path, capsys, processed):
    text = GOOD.replace("[5]", "[99]")
    code = cli.main(
        [
            "questions",
            str(_write(tmp_path, text)),
            "--metadata",
            str(_metadata(tmp_path)),
            "--processed",
            str(processed),
        ]
    )
    assert code == 1 and "page 99 of 1706.03762 doesn't exist" in capsys.readouterr().err
    code = cli.main(
        [
            "questions",
            str(_write(tmp_path, GOOD)),
            "--metadata",
            str(_metadata(tmp_path, ids=("2005.11401",))),
            "--processed",
            str(tmp_path),
        ]
    )
    assert code == 1 and "1706.03762 is not in the corpus" in capsys.readouterr().err


def test_committed_question_file_is_valid(capsys):
    """The real answer key in eval/questions.yaml must always pass (checked in CI)."""
    root = Path(__file__).parent.parent
    code = cli.main(
        [
            "questions",
            str(root / "eval" / "questions.yaml"),
            "--metadata",
            str(root / "data" / "metadata.jsonl"),
            "--processed",
            str(root / "data" / "processed"),
        ]
    )
    assert code == 0, capsys.readouterr().err
