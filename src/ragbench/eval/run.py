"""Run every method on every question and save everything: `ragbench run`.

    ragbench run                                  # keyword, bm25, vector, hybrid
    ragbench run --methods bm25,agent -k 10       # the agent costs money: opt in
    ragbench run --questions q001,q006            # a quick partial run
    ragbench run --methods bm25,agent --judge     # + answers graded by an LLM judge

Each run writes a new folder, never touching an earlier one:

    results/2026-10-26T101500Z/
      run.json          what was run: config, inputs (with checksums), code version,
                        package versions, models, and a per-method summary
      bm25.jsonl        one row per question: ranked results, retrieval metrics, timing
      agent.jsonl       ... plus the agent's answer, stop reason, tokens and cost
                        (with --judge, every row also gets "answer" and "judge":
                        the answer, its tokens, time and cost, and the verdict)
      transcripts/      full agent conversations (not committed: they contain paper text)

A folder describes itself: anyone can see which commit, question set, chunks and models
produced it. `config_hash` identifies the experiment: two runs with the same hash ran the
same methods with the same settings on the same inputs, so their numbers are comparable
(the agent may still differ between runs: the model is not deterministic).

Without --judge a run only scores retrieval and costs nothing (except the agent). With
it, the chunk-based methods get answers written from their top-k passages and every
answer is graded against the reference; see grade.py for how and for the judge's
known pitfalls.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ragbench.eval.grade import VERDICTS, Grader, Judgement
from ragbench.eval.metrics import (
    RETRIEVAL_TYPES,
    hits_from_results,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from ragbench.eval.questions import Question
from ragbench.search.base import Searcher, SearchResult

FORMAT_VERSION = 1
KS = (1, 3, 5, 10)  # recall@k reported for each of these that is <= the run's k
AGENT_DEPTH = 1_000  # keep all the agent's evidence: it has no natural "top k"
PREVIEW_CHARS = 100
PACKAGES = (
    "rag-vs-agent-bench",
    "numpy",
    "pypdf",
    "pyyaml",
    "rank-bm25",
    "sentence-transformers",
    "transformers",
    "torch",
    "anthropic",
)
# settings worth recording, if a searcher has them
_SETTINGS = (
    "k1",
    "b",
    "idf_variant",
    "epsilon",
    "depth",
    "rrf_k",
    "model",
    "max_turns",
    "max_seconds",
    "max_tokens",
)


# --------------------------------------------------------------------------- #
# Describing the environment
# --------------------------------------------------------------------------- #


def git_info(cwd: Path | None = None) -> dict[str, Any] | None:
    """The commit being run, and whether tracked files had uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return None  # not a git checkout, or git not installed


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(searcher: Searcher) -> dict[str, Any]:
    """The searcher's class and settings, including models and fused sub-searchers."""
    info: dict[str, Any] = {"class": type(searcher).__name__}
    for attr in _SETTINGS:
        value = getattr(searcher, attr, None)
        if isinstance(value, str | int | float):
            info[attr] = value
    embedder = getattr(searcher, "embedder", None)
    if embedder is not None:
        info["embedder"] = embedder.name  # includes the pinned revision
    subs = getattr(searcher, "searchers", None)
    if subs:
        info["fuses"] = [describe(s) for s in subs]
    return info


def config_hash(config: Mapping[str, Any]) -> str:
    """Short fingerprint of everything that decides the results."""
    text = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def new_run_dir(root: Path, now: datetime) -> Path:
    """results/<UTC time>; a suffix if two runs start in the same second."""
    base = now.astimezone(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    path, n = root / base, 1
    while path.exists():
        n += 1
        path = root / f"{base}-{n}"
    path.mkdir(parents=True)
    return path


# --------------------------------------------------------------------------- #
# One question, one method
# --------------------------------------------------------------------------- #


def _result_row(rank: int, r: SearchResult, paper: str) -> dict[str, Any]:
    preview = " ".join(r.text.split())
    return {
        "rank": rank,
        "paper": paper,
        "doc_id": r.doc_id,
        "page": r.page,
        "page_end": r.page_end,
        "start": r.start,
        "end": r.end,
        "score": r.score,
        "chunk_id": r.chunk_id,
        "preview": preview[:PREVIEW_CHARS],
    }


def score(results: list[SearchResult], question: Question, k: int, arxiv_of) -> dict | None:
    """Retrieval metrics for one question; None for types without evidence to find."""
    if question.type not in RETRIEVAL_TYPES:
        return None
    hits = hits_from_results(results, arxiv_of)
    metrics: dict[str, float | None] = {}
    for n in (n for n in KS if n <= k):
        metrics[f"recall@{n}"] = recall_at_k(hits, question.gold, n)
    metrics["recall@all"] = recall_at_k(hits, question.gold)  # every result returned
    metrics["paper_recall@all"] = recall_at_k(hits, question.gold, level="paper")
    metrics["rr"] = reciprocal_rank(hits, question.gold)
    metrics["paper_rr"] = reciprocal_rank(hits, question.gold, level="paper")
    return metrics


def _agent_details(run: Any) -> dict[str, Any]:
    from ragbench.search.agent_search import estimate_cost

    skip = {"question", "messages", "reads", "evidence"}  # evidence is in "results"
    details = {k: v for k, v in asdict(run).items() if k not in skip}
    details["cost_usd"] = estimate_cost(run.model, run.input_tokens, run.output_tokens)
    return details


def run_one(
    searcher: Searcher,
    question: Question,
    k: int,
    arxiv_of: Mapping[str, str],
    grader: Grader | None = None,
) -> dict[str, Any]:
    """Search, time it, score it; with a grader, also answer and judge.

    A failing search is recorded, never raised.
    """
    is_agent = hasattr(searcher, "last_run")
    if is_agent:
        searcher.last_run = None  # so a failed call can't report the previous run
    error = ""
    started = time.perf_counter()
    try:
        results = searcher.search(question.question, k=AGENT_DEPTH if is_agent else k)
    except Exception as exc:  # one broken question must not end a whole benchmark
        results, error = [], f"{type(exc).__name__}: {exc}"
    seconds = round(time.perf_counter() - started, 3)

    row: dict[str, Any] = {
        "id": question.id,
        "type": question.type,
        "question": question.question,
        "author": question.author,
        "method": searcher.name,
        "k": k,
        "seconds": seconds,
        "error": error,
        "metrics": score(results, question, k, arxiv_of),
        "results": [
            _result_row(i, r, arxiv_of.get(r.doc_id, r.doc_id))
            for i, r in enumerate(results, start=1)
        ],
    }
    run = getattr(searcher, "last_run", None)
    if run is not None:
        row["agent"] = _agent_details(run)
        if run.error and not error:
            row["error"] = run.error
    if grader is not None:
        row["answer"], row["judge"] = _answer_and_judge(grader, question, results[:k], row)
    return row


def _answer_and_judge(
    grader: Grader, question: Question, results: list[SearchResult], row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The method's answer (the agent's own, or generated from the passages) + verdict."""
    if "agent" in row:
        a = row["agent"]
        answer = {
            "source": "agent",
            "model": a["model"],
            "text": a["answer"],
            "stop": a["stop"],
            "input_tokens": a["input_tokens"],
            "output_tokens": a["output_tokens"],
            "seconds": row["seconds"],
            "cost_usd": a["cost_usd"],
            "error": a["error"],
        }
    elif row["error"]:  # the search itself failed: the method gave no answer
        answer = {"source": "none", "text": "", "cost_usd": 0.0, "error": row["error"]}
    else:
        answer = {"source": "generated", **grader.answer(question, results).to_json()}

    if answer["source"] == "generated" and answer["error"]:
        # our own answering call failed: that says nothing about the method
        judgement = Judgement(None, "not judged: the answer could not be generated", None)
    else:
        judgement = grader.judge(question, answer["text"])
    return answer, judgement.to_json()


# --------------------------------------------------------------------------- #
# The whole benchmark
# --------------------------------------------------------------------------- #


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-method totals: mean metrics over the questions that have them."""
    scored = [r["metrics"] for r in rows if r["metrics"] is not None]
    keys = list(dict.fromkeys(key for m in scored for key in m))
    summary: dict[str, Any] = {
        "questions": len(rows),
        "scored": len(scored),
        "errors": sum(bool(r["error"]) for r in rows),
        "seconds": round(sum(r["seconds"] for r in rows), 3),
        "metrics": {key: mean(m.get(key) for m in scored) for key in keys},
    }
    judged = [r for r in rows if "judge" in r]
    if judged:
        verdicts = Counter(r["judge"]["verdict"] or "unjudged" for r in judged)
        by_type: dict[str, list[float | None]] = {}
        by_author: dict[str, list[float | None]] = {}
        for r in judged:
            by_type.setdefault(r["type"], []).append(r["judge"]["points"])
            by_author.setdefault(r.get("author") or "unknown", []).append(r["judge"]["points"])
        summary["answers"] = {
            # points: F1 for list questions, else correct 1 / partial 0.5 / wrong 0
            "accuracy": mean(r["judge"]["points"] for r in judged),
            "by_type": {t: mean(points) for t, points in by_type.items()},
            "by_author": {a: mean(points) for a, points in by_author.items()},
            **{v: verdicts.get(v, 0) for v in (*VERDICTS, "unjudged")},
            "abstained": sum(r["judge"].get("abstained") is True for r in judged),
            "confidently_wrong": sum(bool(r["judge"].get("confidently_wrong")) for r in judged),
            "needs_review": [r["id"] for r in judged if r["judge"].get("needs_review")],
            "answer_cost_usd": _total(r["answer"].get("cost_usd") for r in judged),
            # search + writing the answer; the agent's search time already includes it
            "seconds_to_answer": round(
                sum(
                    r["seconds"]
                    + (r["answer"]["seconds"] if r["answer"]["source"] == "generated" else 0)
                    for r in judged
                ),
                3,
            ),
            "judge_cost_usd": _total(r["judge"].get("cost_usd") for r in judged),
        }
    agent = [r["agent"] for r in rows if "agent" in r]
    if agent:
        costs = [a["cost_usd"] for a in agent]
        summary["agent"] = {
            "input_tokens": sum(a["input_tokens"] for a in agent),
            "output_tokens": sum(a["output_tokens"] for a in agent),
            "cost_usd": None if None in costs else round(sum(costs), 4),
            "stops": dict(Counter(a["stop"] for a in agent)),
        }
    return summary


def _total(costs) -> float | None:
    """Sum of costs; None if any is unknown (a model without a price), never a guess."""
    costs = list(costs)
    return None if None in costs else round(sum(costs), 4)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_benchmark(
    questions: Sequence[Question],
    searchers: Mapping[str, Searcher],
    *,
    k: int,
    arxiv_of: Mapping[str, str],
    out_dir: Path,
    info: Mapping[str, Any],
    grader: Grader | None = None,
    progress: Callable[[str], None] = lambda line: None,
) -> dict[str, Any]:
    """Run each searcher on each question, writing rows as they finish.

    `info` is the run's description (config, inputs, environment). run.json is written
    before the first question (status "running") and again at the end ("finished", or
    "interrupted" after Ctrl-C), so even a crashed run says what it was.
    """
    run_info: dict[str, Any] = {
        "format": FORMAT_VERSION,
        **info,
        "status": "running",
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished": None,
        "summary": {},
    }
    _write_json(out_dir / "run.json", run_info)
    try:
        for name, searcher in searchers.items():
            rows = []
            with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for question in questions:
                    row = run_one(searcher, question, k, arxiv_of, grader)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()  # a crash keeps every finished row
                    rows.append(row)
                    flag = " ERROR" if row["error"] else ""
                    if "judge" in row:
                        flag += f" {row['judge']['verdict'] or 'unjudged'}"
                    progress(f"{name} {question.id} {row['seconds']:.2f}s{flag}")
            run_info["summary"][name] = summarize(rows)
        run_info["status"] = "finished"
    except KeyboardInterrupt:
        run_info["status"] = "interrupted"
        raise
    finally:
        run_info["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
        _write_json(out_dir / "run.json", run_info)
    return run_info


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #

DEFAULT_METHODS = ("keyword", "bm25", "vector", "hybrid")  # the agent is opt-in: it costs money


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def select_questions(questions: Sequence[Question], ids: list[str]) -> list[Question]:
    if not ids:
        return list(questions)
    known = {q.id for q in questions}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown question id(s): {', '.join(unknown)}")
    wanted = set(ids)
    return [q for q in questions if q.id in wanted]  # file order, not argument order


def main_run(args: Any) -> int:
    """`ragbench run`: build the methods, describe the run, run it, print a summary."""
    from ragbench.cli import METHODS, build_searchers, find_chunks_file
    from ragbench.eval.questions import QuestionSetError, load

    methods = _split(args.methods) or list(DEFAULT_METHODS)
    bad = [m for m in methods if m not in METHODS]
    if bad:
        print(
            f"error: unknown method(s) {', '.join(bad)}; choose from {', '.join(METHODS)}",
            file=sys.stderr,
        )
        return 1
    methods = list(dict.fromkeys(methods))
    manifest = args.processed / "manifest.jsonl"
    if not manifest.exists():
        print(f"error: {manifest} not found; run ragbench-ingest first", file=sys.stderr)
        return 1
    try:
        question_set = load(args.question_file)
        questions = select_questions(question_set.questions, _split(args.questions))
        chunks_path = (
            find_chunks_file(args.processed, args.chunks) if set(methods) - {"agent"} else None
        )
    except (FileNotFoundError, ValueError) as exc:
        problems = exc.problems if isinstance(exc, QuestionSetError) else [str(exc)]
        print("\n".join(f"error: {p}" for p in problems), file=sys.stderr)
        return 1

    out_dir = new_run_dir(args.out, datetime.now(UTC))
    try:
        searchers = build_searchers(
            methods,
            args.processed,
            chunks=chunks_path,
            model=args.model,
            max_turns=args.max_turns,
            transcript_dir=out_dir / "transcripts",
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        out_dir.rmdir()  # nothing was run
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    arxiv_of = {m["doc_id"]: m.get("arxiv_id") or m["doc_id"] for m in entries}
    grader = None
    if args.judge:
        from ragbench.eval import grade

        try:
            client = grade.default_client()
        except ImportError as exc:
            out_dir.rmdir()
            print(f"error: {exc}", file=sys.stderr)
            return 1
        papers = {m["doc_id"]: (arxiv_of[m["doc_id"]], m["title"]) for m in entries}
        # answers are written by the agent's model, so only retrieval differs
        grader = grade.Grader(client, papers, answer_model=args.model, judge_model=args.judge_model)
    inputs = {
        "questions": {
            "path": str(args.question_file),
            "sha256": sha256_file(args.question_file),
            "ids": [q.id for q in questions],
        },
        "manifest_sha256": sha256_file(manifest),
        "chunks": (
            {"file": chunks_path.name, "sha256": sha256_file(chunks_path)} if chunks_path else None
        ),
    }
    config = {
        "methods": methods,
        "k": args.k,
        "searchers": {name: describe(s) for name, s in searchers.items()},
        "grader": grader.describe() if grader else None,
    }
    info = {
        "config": config,
        "inputs": inputs,
        "config_hash": config_hash({"config": config, "inputs": inputs}),
        "command": ["ragbench", *sys.argv[1:]],
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    print(f"run {out_dir.name}: {len(methods)} method(s) x {len(questions)} question(s)")
    try:
        run_info = run_benchmark(
            questions,
            searchers,
            k=args.k,
            arxiv_of=arxiv_of,
            out_dir=out_dir,
            info=info,
            grader=grader,
            progress=lambda line: print(f"  {line}", file=sys.stderr),
        )
    except KeyboardInterrupt:
        print(f"\ninterrupted; partial results in {out_dir}", file=sys.stderr)
        return 130
    print("\n".join(format_summary(run_info["summary"], args.k)))
    print(f"\nsaved to {out_dir}")
    return 0


def format_summary(summary: Mapping[str, Any], k: int) -> list[str]:
    """A small table: one line per method."""
    ks = [n for n in KS if n <= k]
    cols = [f"recall@{n}" for n in ks] + ["paper_recall@all", "rr"]
    judged = any("answers" in s for s in summary.values())
    if judged:
        cols.append("accuracy")
    head = f"{'method':<8} " + " ".join(f"{c:>10}" for c in cols) + f" {'errors':>6} {'time':>8}"
    lines = [head.replace("paper_recall@all", "paper_r@all")]
    for name, s in summary.items():
        vals = [s["metrics"].get(c) for c in cols if c != "accuracy"]
        if judged:
            vals.append(s.get("answers", {}).get("accuracy"))
        cells = " ".join(f"{v:>10.3f}" if v is not None else f"{'-':>10}" for v in vals)
        seconds = s["answers"]["seconds_to_answer"] if "answers" in s else s["seconds"]
        line = f"{name:<8} {cells} {s['errors']:>6} {seconds:>7.1f}s"
        cost = (
            s["answers"]["answer_cost_usd"]
            if "answers" in s
            else s.get("agent", {}).get("cost_usd", 0.0)
        )
        if "answers" in s or "agent" in s:
            line += f"  cost≈${cost:.3f}" if cost is not None else "  cost unknown"
        lines.append(line)
    if judged:
        costs = [s["answers"]["judge_cost_usd"] for s in summary.values() if "answers" in s]
        total = None if None in costs else sum(costs)
        lines.append(
            "accuracy: F1 of papers named for list questions, else correct 1, partial 0.5, "
            "wrong 0; "
            "time: search + answer; "
            + (f"judging cost ${total:.3f}" if total is not None else "judging cost unknown")
        )
        review = [
            f"{name} {qid}"
            for name, s in summary.items()
            for qid in s.get("answers", {}).get("needs_review", [])
        ]
        if review:
            lines.append(f"check by hand (verdict and F1 disagree): {', '.join(review)}")
    return lines
