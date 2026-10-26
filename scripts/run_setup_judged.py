"""`ragbench run --judge` for the retrieval sweep's setups (stronger embedders, reranker).

    python scripts/run_setup_judged.py hybrid-bge+rerank            # answers + judging
    python scripts/run_setup_judged.py hybrid-bge+rerank --no-judge # retrieval only, free

`ragbench run` only knows the first run's methods (bm25, vector, hybrid, ...). This builds
a searcher from eval/sweep.py's CONFIGS instead and hands it to the same run_benchmark, so
the run folder (run.json, <setup>.jsonl) has the usual format: `ragbench summarize` and
`ragbench stats` read it like any other run. Answers are written by the same model as in
the committed run, from the top k = 10 passages, and judged by the same judge.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from ragbench.eval.questions import load
from ragbench.eval.run import (
    _split,
    config_hash,
    describe,
    format_summary,
    git_info,
    new_run_dir,
    package_versions,
    run_benchmark,
    select_questions,
    sha256_file,
)
from ragbench.eval.sweep import CONFIGS, Builder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("setup", choices=sorted(CONFIGS))
    parser.add_argument("--questions", help="comma-separated question ids (default: all)")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--question-file", type=Path, default=Path("eval/questions.yaml"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--model", default="claude-sonnet-5-5", help="writes the answers")
    parser.add_argument("--judge-model", default="claude-opus-5-5")
    parser.add_argument("--no-judge", action="store_true", help="retrieval only, no API calls")
    args = parser.parse_args()

    manifest = args.processed / "manifest.jsonl"
    questions = select_questions(load(args.question_file).questions, _split(args.questions))
    searcher = Builder(args.processed).build(args.setup, CONFIGS[args.setup])
    searchers = {args.setup: searcher}

    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    arxiv_of = {m["doc_id"]: m.get("arxiv_id") or m["doc_id"] for m in entries}
    grader = None
    if not args.no_judge:
        from ragbench.eval import grade

        papers = {m["doc_id"]: (arxiv_of[m["doc_id"]], m["title"]) for m in entries}
        grader = grade.Grader(
            grade.default_client(), papers, answer_model=args.model, judge_model=args.judge_model
        )

    chunking = CONFIGS[args.setup].chunks
    inputs = {
        "questions": {
            "path": str(args.question_file),
            "sha256": sha256_file(args.question_file),
            "ids": [q.id for q in questions],
        },
        "manifest_sha256": sha256_file(manifest),
        "chunks": {"setup_chunking": chunking},
    }
    config = {
        "methods": [args.setup],
        "k": args.k,
        "searchers": {args.setup: describe(searcher)},
        "grader": grader.describe() if grader else None,
        "sweep_setup": {"name": args.setup, **vars(CONFIGS[args.setup])},
    }
    info = {
        "config": config,
        "inputs": inputs,
        "config_hash": config_hash({"config": config, "inputs": inputs}),
        "command": ["scripts/run_setup_judged.py", *sys.argv[1:]],
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    out_dir = new_run_dir(args.out, datetime.now(UTC))
    print(f"run {out_dir.name}: 1 method(s) x {len(questions)} question(s)")
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
    return 0 if run_info["status"] == "finished" else 2


if __name__ == "__main__":
    sys.exit(main())
