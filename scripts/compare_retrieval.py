"""Judged retrieval runs on the current question set, next to BM25 and the agent (no API calls).

    python scripts/compare_retrieval.py [results/retrieval-judged-runs.txt]
                                        [results/model-comparison-runs.txt]

Reads the runs listed by scripts/run_retrieval_judged.sh and, for reference, the agent
runs listed by scripts/run_model_comparison.sh. Reports accuracy overall and by question
type, and paired bootstrap differences (same method as `ragbench stats`). Questions are
compared on their points averaged over a setup's runs.

Writes results/retrieval-judged.md and prints it.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from ragbench.eval.stats import bootstrap_mean, paired
from ragbench.eval.summarize import load_run

TYPES = ["single_fact", "multi_filter", "cross_doc", "negation", "unanswerable"]


def read_list(path: Path) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _repeat, run_dir = line.split()
                # listed relative to the repo root, which holds results/
                runs[name].append(path.parent.parent / run_dir)
    return runs


def rows_of(run_dir: Path) -> list[dict]:
    _info, rows = load_run(run_dir)
    if len(rows) != 1:
        raise SystemExit(f"{run_dir}: expected one method, found {sorted(rows)}")
    return next(iter(rows.values()))


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def main() -> int:
    retrieval_list = Path(sys.argv[1] if len(sys.argv) > 1 else "results/retrieval-judged-runs.txt")
    agent_list = Path(sys.argv[2] if len(sys.argv) > 2 else "results/model-comparison-runs.txt")
    runs = dict(read_list(retrieval_list))
    for model, dirs in read_list(agent_list).items():
        runs[f"agent ({model})"] = dirs
    if not runs:
        print("no runs listed yet")
        return 1

    setups = {name: [rows_of(d) for d in dirs] for name, dirs in runs.items()}
    per_question: dict[str, dict[str, float]] = {}
    out = [
        "### Judged runs on the current question set",
        "",
        "| setup | runs | accuracy per run | mean | 95% interval | recall@5 | MRR | cost / q |",
        "|:---|---:|:---|---:|:---:|---:|---:|---:|",
    ]
    for name, runs_rows in setups.items():
        points, accs, r5, rr, cost = defaultdict(list), [], [], [], []
        for rows in runs_rows:
            judged = [r for r in rows if (r.get("judge") or {}).get("points") is not None]
            accs.append(mean(float(r["judge"]["points"]) for r in judged))
            for r in judged:
                points[r["id"]].append(float(r["judge"]["points"]))
            for r in rows:
                m = r.get("metrics") or {}
                if m.get("recall@5") is not None:
                    r5.append(m["recall@5"])
                if m.get("rr") is not None:
                    rr.append(m["rr"])
                # an agent row repeats its cost under "answer"; count it once
                part = "agent" if r.get("agent") else "answer"
                cost.append((r.get(part) or {}).get("cost_usd") or 0.0)
        per_question[name] = {q: mean(p) for q, p in points.items()}
        m, lo, hi = bootstrap_mean(list(per_question[name].values()))
        out.append(
            f"| {name} | {len(runs_rows)} | {', '.join(fmt(a) for a in accs)} | {fmt(m)} | "
            f"{fmt(lo)} to {fmt(hi)} | {fmt(mean(r5) if r5 else None)} | "
            f"{fmt(mean(rr) if rr else None)} | ${mean(cost):.3f} |"
        )

    out += [
        "",
        "#### Accuracy by question type (mean over runs)",
        "",
        "| question type | " + " | ".join(setups) + " |",
        "|:---|" + "---:|" * len(setups),
    ]
    for qtype in TYPES:
        cells = []
        for runs_rows in setups.values():
            vals = [
                float(r["judge"]["points"])
                for rows in runs_rows
                for r in rows
                if r["type"] == qtype and (r.get("judge") or {}).get("points") is not None
            ]
            cells.append(fmt(mean(vals)) if vals else "–")
        n = sum(1 for r in next(iter(setups.values()))[0] if r["type"] == qtype)
        out.append(f"| {qtype} ({n}) | " + " | ".join(cells) + " |")

    out += [
        "",
        "#### Paired differences",
        "",
        "| comparison | difference | 95% interval | better / same / worse | clear? |",
        "|:---|---:|:---:|:---:|:---:|",
    ]
    for a, b in combinations(per_question, 2):
        p = paired(per_question[a], per_question[b])
        clear = "yes" if p["low"] is not None and (p["low"] > 0 or p["high"] < 0) else "no"
        out.append(
            f"| {a} vs {b} | {fmt(p['diff'], True)} | {fmt(p['low'], True)} to "
            f"{fmt(p['high'], True)} | {p['wins']} / {p['ties']} / {p['losses']} | {clear} |"
        )

    out += [
        "",
        "Runs: " + "; ".join(f"{n}: {', '.join(d.name for d in ds)}" for n, ds in runs.items()),
    ]
    text = "\n".join(out) + "\n"
    (retrieval_list.parent / "retrieval-judged.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
