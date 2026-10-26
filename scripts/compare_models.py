"""Compare the agent across models and repeated runs (no API calls).

    python scripts/compare_models.py results/model-comparison-runs.txt

The list file has one line per run: "<model> <repeat> <run dir>", as written by
scripts/run_model_comparison.sh. For each model this reports accuracy per run (so the
run-to-run spread is visible), accuracy by question type averaged over its runs, how runs
ended, and cost. Models are compared with the same paired bootstrap as `ragbench stats`,
on each question's points averaged over that model's runs.

Writes <list name>.md next to the list file (model-comparison.md) and prints it.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from ragbench.eval.stats import bootstrap_mean, paired
from ragbench.eval.summarize import load_run

TYPES = ["single_fact", "multi_filter", "cross_doc", "negation", "unanswerable"]


def read_list(path: Path) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            model, _repeat, run_dir = line.split()
            # listed relative to the repo root, which holds results/
            runs[model].append(path.parent.parent / run_dir)
    return runs


def agent_rows(run_dir: Path) -> list[dict]:
    _info, rows = load_run(run_dir)
    if "agent" not in rows:
        raise SystemExit(f"{run_dir}: no agent.jsonl")
    return rows["agent"]


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def render(runs: dict[str, list[Path]]) -> str:
    rows_by_model = {m: [agent_rows(d) for d in dirs] for m, dirs in runs.items()}
    out = ["### Agent by model", ""]

    out += [
        "| model | runs | accuracy per run | mean | 95% interval | max_turns stops | "
        "cost / q | time / q |",
        "|:---|---:|:---|---:|:---:|---:|---:|---:|",
    ]
    per_question: dict[str, dict[str, float]] = {}
    for model, runs_rows in rows_by_model.items():
        accs, stops, costs, secs = [], Counter(), [], []
        points = defaultdict(list)
        for rows in runs_rows:
            judged = [r for r in rows if (r.get("judge") or {}).get("points") is not None]
            accs.append(mean(r["judge"]["points"] for r in judged))
            for r in rows:
                stops[(r.get("agent") or {}).get("stop", "?")] += 1
                costs.append((r.get("agent") or {}).get("cost_usd") or 0.0)
                secs.append(r.get("seconds") or 0.0)
            for r in judged:
                points[r["id"]].append(float(r["judge"]["points"]))
        per_question[model] = {q: mean(p) for q, p in points.items()}
        m, lo, hi = bootstrap_mean(list(per_question[model].values()))
        out.append(
            f"| {model} | {len(runs_rows)} | {', '.join(fmt(a) for a in accs)} | {fmt(m)} | "
            f"{fmt(lo)} to {fmt(hi)} | {stops.get('max_turns', 0)} | ${mean(costs):.3f} | "
            f"{mean(secs):.1f} s |"
        )

    out += [
        "",
        "#### Accuracy by question type (mean over runs)",
        "",
        "| question type | " + " | ".join(rows_by_model) + " |",
        "|:---|" + "---:|" * len(rows_by_model),
    ]
    for qtype in TYPES:
        cells = []
        for runs_rows in rows_by_model.values():
            vals = [
                float(r["judge"]["points"])
                for rows in runs_rows
                for r in rows
                if r["type"] == qtype and (r.get("judge") or {}).get("points") is not None
            ]
            cells.append(fmt(mean(vals)) if vals else "–")
        n = sum(1 for r in next(iter(rows_by_model.values()))[0] if r["type"] == qtype)
        out.append(f"| {qtype} ({n}) | " + " | ".join(cells) + " |")

    out += [
        "",
        "#### Paired differences (points averaged over each model's runs)",
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
        "Run folders: "
        + "; ".join(f"{m}: {', '.join(d.name for d in ds)}" for m, ds in runs.items()),
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    list_file = Path(sys.argv[1])
    text = render(read_list(list_file))
    (list_file.parent / f"{list_file.stem.removesuffix('-runs')}.md").write_text(
        text, encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
