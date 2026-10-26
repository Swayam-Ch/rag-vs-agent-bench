"""Turn a results folder into markdown tables: `ragbench summarize results/<run>`.

    ragbench summarize results/2026-10-26T120551Z > summary.md

The numbers are recomputed from the per-question rows (<method>.jsonl) with the same
code the runner uses, not copied from run.json, so the tables can't drift from the raw
data. The output pastes straight into the README:

* the main comparison: one row per method (retrieval, accuracy, cost, latency)
* accuracy by question type, and by who wrote the question
* what was run: commit, models, question set, and any problems (errors, unjudged)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragbench.eval.questions import TYPES
from ragbench.eval.run import summarize as summarize_rows

DASH = "–"


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """run.json and the rows of every method, in the order the run used."""
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    methods = info.get("config", {}).get("methods") or sorted(
        p.stem for p in run_dir.glob("*.jsonl")
    )
    rows = {}
    for method in methods:
        path = run_dir / f"{method}.jsonl"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            rows[method] = [json.loads(line) for line in lines if line.strip()]
    return info, rows


def _num(value: float | None, digits: int = 2) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:.4f}" if value < 0.01 else f"${value:.3f}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> list[str]:
    """A markdown table; `align` has one character per column: l(eft) or r(ight)."""
    marks = {"l": ":---", "r": "---:"}
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def main_table(summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Method x (recall@5, MRR, paper recall, accuracy, cost and latency per question)."""
    rows = []
    for method, s in summaries.items():
        m, a = s["metrics"], s.get("answers", {})
        n = s["questions"] or 1
        if a:
            cost = a["answer_cost_usd"]
            seconds = a["seconds_to_answer"]
        else:  # retrieval-only run: the agent still has a cost, the others don't
            cost = s.get("agent", {}).get("cost_usd", 0.0)
            seconds = s["seconds"]
        rows.append(
            [
                f"**{method}**",
                _num(m.get("recall@5")),
                _num(m.get("rr")),
                _num(m.get("paper_recall@all")),
                _num(a.get("accuracy")),
                _money(None if cost is None else cost / n),
                f"{seconds / n:.1f} s",
            ]
        )
    head = ["method", "recall@5", "MRR", "paper recall", "accuracy", "cost / q", "time / q"]
    return _table(head, rows, "lrrrrrr")


def breakdown_table(summaries: dict[str, dict[str, Any]], key: str, label: str) -> list[str]:
    """Accuracy by question type (key="by_type") or by author (key="by_author")."""
    groups: list[str] = []
    for s in summaries.values():
        for g in s.get("answers", {}).get(key, {}):
            if g not in groups:
                groups.append(g)
    if key == "by_type":  # the format's own order, not the order seen first
        groups = [t for t in TYPES if t in groups]
    counts = _group_counts(summaries, key)
    rows = []
    for g in groups:
        cells = [_num(s.get("answers", {}).get(key, {}).get(g)) for s in summaries.values()]
        rows.append([f"{g} ({counts.get(g, 0)})", *cells])
    return _table([label, *summaries], rows, "l" + "r" * len(summaries))


def _group_counts(summaries: dict[str, dict[str, Any]], key: str) -> dict[str, int]:
    """Questions per group, from the first method's rows (all methods answer the same)."""
    rows = next(iter(summaries.values()), {}).get("_rows", [])
    field = "type" if key == "by_type" else "author"
    out: dict[str, int] = {}
    for r in rows:
        group = r.get(field) or "unknown"
        out[group] = out.get(group, 0) + 1
    return out


def run_details(info: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> list[str]:
    git = info.get("git") or {}
    commit = (git.get("commit") or "unknown")[:7] + (
        " (uncommitted changes!)" if git.get("dirty") else ""
    )
    questions = info.get("inputs", {}).get("questions", {})
    grader = info.get("config", {}).get("grader")
    lines = [
        f"- run: `{info.get('started', '?')}`, status **{info.get('status', '?')}**, "
        f"commit `{commit}`, config `{info.get('config_hash', '?')}`",
        f"- questions: {len(questions.get('ids', []))} from `{questions.get('path', '?')}`, "
        f"k = {info.get('config', {}).get('k', '?')}",
    ]
    searchers = info.get("config", {}).get("searchers", {})
    embedder = next(
        (
            s.get("embedder")
            or next((f.get("embedder") for f in s.get("fuses", []) if f.get("embedder")), None)
            for s in searchers.values()
            if s.get("embedder") or s.get("fuses")
        ),
        None,
    )
    if embedder:
        lines.append(f"- embeddings: `{embedder}`")
    if "agent" in searchers:
        a = searchers["agent"]
        lines.append(f"- agent: `{a.get('model')}`, at most {a.get('max_turns')} turns")
    if grader:
        judge_cost = [s["answers"]["judge_cost_usd"] for s in summaries.values() if "answers" in s]
        total = None if None in judge_cost else sum(judge_cost)
        lines.append(
            f"- answers written by `{grader['answer_model']}` from the top-k passages "
            f"(agent: its own); judged by `{grader['judge_model']}` ({grader['judge_prompt']}), "
            f"judging cost {_money(total)}"
        )
    problems = []
    for method, s in summaries.items():
        a = s.get("answers", {})
        if s["errors"]:
            problems.append(f"{method}: {s['errors']} errors")
        if a.get("unjudged"):
            problems.append(f"{method}: {a['unjudged']} unjudged")
        if a.get("needs_review"):
            problems.append(f"{method}: check {', '.join(a['needs_review'])} by hand")
    lines.append("- problems: " + ("; ".join(problems) if problems else "none"))
    return lines


def render(run_dir: Path) -> str:
    info, rows = load_run(run_dir)
    if not rows:
        raise FileNotFoundError(f"no <method>.jsonl files in {run_dir}")
    summaries = {}
    for method, method_rows in rows.items():
        summaries[method] = summarize_rows(method_rows)
        summaries[method]["_rows"] = method_rows
    judged = any("answers" in s for s in summaries.values())

    out = [f"### Results: `{run_dir.name}`", ""]
    out += main_table(summaries)
    out += [
        "",
        "recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with "
        "evidence to find). Paper recall: share of gold papers found in everything a method "
        "returned. Accuracy: judged answers; F1 of the papers named for list questions, else "
        "correct 1, partial 0.5, wrong 0. Cost and time per question include writing the "
        "answer, not judging it.",
    ]
    if judged:
        out += ["", "#### Accuracy by question type", ""]
        out += breakdown_table(summaries, "by_type", "question type")
        if any(len(s.get("answers", {}).get("by_author", {})) > 1 for s in summaries.values()):
            out += ["", "#### Accuracy by question author", ""]
            out += breakdown_table(summaries, "by_author", "written by")
    out += ["", "#### Run details", ""]
    out += run_details(info, summaries)
    return "\n".join(out) + "\n"
