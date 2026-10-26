"""How sure are the numbers? `ragbench stats results/<run>`

Two checks that need no API calls, only the rows a judged run already saved:

1. **Confidence intervals.** With 20 questions, one question moves accuracy by 0.05. A
   bootstrap resamples the questions with replacement many times and recomputes the
   numbers; the middle 95% of the results is the interval. For two methods it resamples
   the *same* questions for both (a paired bootstrap), because they answered the same
   questions: if the interval of the difference excludes 0, the difference is not just
   the luck of which 20 questions were asked.

2. **The agent's citations.** The agent ends each answer with "Sources: <paper> p.<page>".
   Each citation is checked against the gold evidence, and against what the agent was
   actually shown (its grep snippets and read_file ranges). A citation of a page it never
   saw came from memory, not from searching.

Writes stats.md into the run folder and prints it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.eval.summarize import load_run

RESAMPLES = 10_000
SEED = 0  # fixed, so the intervals are the same every time the command runs
_ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_mean(values: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED):
    """(mean, low, high): the mean and its 95% percentile-bootstrap interval."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(low), float(high)


def points_by_question(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Question id -> points, for judged questions only (unjudged ones are left out)."""
    out = {}
    for r in rows:
        points = (r.get("judge") or {}).get("points")
        if points is not None:
            out[r["id"]] = float(points)
    return out


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """The difference a - b on the questions both answered, with its interval."""
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    mean, low, high = bootstrap_mean(diffs)
    return {
        "n": len(ids),
        "diff": mean,
        "low": low,
        "high": high,
        "wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #


def paper_ids(rows_by_method: dict[str, list[dict[str, Any]]], processed: Path | None) -> dict:
    """doc_id -> arXiv ID, from the run's own results (and the manifest, if available)."""
    out: dict[str, str] = {}
    for rows in rows_by_method.values():
        for row in rows:
            for r in row.get("results", []):
                if r.get("doc_id") and r.get("paper"):
                    out[r["doc_id"]] = r["paper"]
    manifest = processed / "manifest.jsonl" if processed else None
    if manifest and manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            out.setdefault(m["doc_id"], m.get("arxiv_id") or m["doc_id"])
    return out


def _cited_paper(citation: dict[str, Any], arxiv_of: dict[str, str]) -> str:
    ref = re.sub(r"v\d+$", "", str(citation.get("ref", "")))
    if _ARXIV.match(ref):
        return ref
    return arxiv_of.get(citation.get("doc_id") or ref, ref)


def check_citations(
    row: dict[str, Any], gold: list[dict[str, Any]], arxiv_of: dict[str, str]
) -> dict[str, Any]:
    """Each citation: does it hit gold evidence, and had the agent seen that page?

    `gold`: [{"paper": arXiv ID, "pages": [...]}]; empty pages mean "the whole paper".
    """
    seen = {(r["paper"], p) for r in row["results"] for p in range(r["page"], r["page_end"] + 1)}
    checked = []
    for c in row["agent"].get("citations", []):
        paper, page = _cited_paper(c, arxiv_of), int(c["page"])
        on_gold = any(g["paper"] == paper and (not g["pages"] or page in g["pages"]) for g in gold)
        checked.append(
            {"paper": paper, "page": page, "gold": on_gold, "seen": (paper, page) in seen}
        )
    found = sum(
        any(
            c["paper"] == g["paper"] and (not g["pages"] or c["page"] in g["pages"])
            for c in checked
        )
        for g in gold
    )
    return {
        "citations": checked,
        "gold_entries": len(gold),
        "gold_cited": found,
    }


def gold_by_question(run_info: dict[str, Any], question_file: Path | None) -> dict[str, list]:
    """Question id -> gold entries, from the question file the run used."""
    from ragbench.eval.questions import load

    path = question_file or Path(run_info.get("inputs", {}).get("questions", {}).get("path", ""))
    if not path or not path.exists():
        return {}
    return {
        q.id: [{"paper": e.paper, "pages": list(e.pages)} for e in q.gold]
        for q in load(path).questions
    }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def render(run_dir: Path, question_file: Path | None = None, processed: Path | None = None) -> str:
    info, rows = load_run(run_dir)
    points = {m: points_by_question(r) for m, r in rows.items()}
    methods = [m for m in rows if points[m]]
    if not methods:
        raise ValueError(f"{run_dir} has no judged answers; run it with --judge")

    out = [f"### How sure are the numbers? `{run_dir.name}`", ""]
    out += [
        f"95% bootstrap intervals over the judged questions ({RESAMPLES:,} resamples, "
        f"seed {SEED}). With this few questions, read the intervals, not the decimals.",
        "",
        "| method | accuracy | 95% interval | questions |",
        "|:---|---:|:---:|---:|",
    ]
    for m in methods:
        mean, low, high = bootstrap_mean(list(points[m].values()))
        out.append(f"| {m} | {_fmt(mean)} | {_fmt(low)} to {_fmt(high)} | {len(points[m])} |")

    out += [
        "",
        "Paired differences: both methods on the same questions. *Clear* means the interval "
        "excludes 0.",
        "",
        "| comparison | difference | 95% interval | better / same / worse | clear? |",
        "|:---|---:|:---:|:---:|:---:|",
    ]
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            first, second = (b, a) if b == "agent" else (a, b)  # the agent always first
            p = paired(points[first], points[second])
            clear = p["low"] is not None and (p["low"] > 0 or p["high"] < 0)
            out.append(
                f"| {first} vs {second} | {_fmt(p['diff'], True)} | "
                f"{_fmt(p['low'], True)} to {_fmt(p['high'], True)} | "
                f"{p['wins']} / {p['ties']} / {p['losses']} | {'yes' if clear else 'no'} |"
            )

    if "agent" in rows:
        out += ["", *citation_section(rows, info, question_file, processed)]
    return "\n".join(out) + "\n"


def citation_section(rows, info, question_file, processed) -> list[str]:
    """Citation checks over the agent's rows.

    Unanswerable questions have no gold to cite, so they are skipped. For negation
    questions the "was it shown?" check doesn't apply: the evidence for "paper X never
    mentions Y" is a grep that found nothing, so there is no page to have been shown.
    """
    gold = gold_by_question(info, question_file)
    if not gold:
        return ["(citations not checked: the question file wasn't found)"]
    arxiv_of = paper_ids(rows, processed)
    total = on_gold = other_page = other_paper = 0
    checkable = seen = unseen_gold = 0
    gold_entries = gold_cited = 0
    unseen: list[str] = []
    no_citations: list[str] = []
    for row in rows["agent"]:
        entries = gold.get(row["id"])
        if not entries or "agent" not in row:  # unanswerable, or not in the question file
            continue
        result = check_citations(row, entries, arxiv_of)
        cites = result["citations"]
        if not cites:
            no_citations.append(row["id"])
        total += len(cites)
        on_gold += sum(c["gold"] for c in cites)
        gold_papers = {e["paper"] for e in entries}
        other_page += sum(not c["gold"] and c["paper"] in gold_papers for c in cites)
        other_paper += sum(not c["gold"] and c["paper"] not in gold_papers for c in cites)
        gold_entries += result["gold_entries"]
        gold_cited += result["gold_cited"]
        if row.get("type") == "negation":
            continue
        checkable += len(cites)
        seen += sum(c["seen"] for c in cites)
        for c in cites:
            if not c["seen"]:
                unseen.append(f"{row['id']} ({c['paper']} p.{c['page']})")
                unseen_gold += c["gold"]
    share = lambda n, d: f"{n} of {d} ({n / d:.0%})" if d else "none"  # noqa: E731
    lines = [
        "#### The agent's citations",
        "",
        'Each page in an answer\'s "Sources:" list, checked against the gold evidence and '
        "against the pages the agent was shown (grep snippets and read_file ranges). "
        "Unanswerable questions are skipped; negation questions are left out of the "
        '"shown" check, because evidence of absence is a grep with no matches.',
        "",
        f"- citations on a gold page: {share(on_gold, total)}; of the rest, {other_page} "
        f"are other pages of a gold paper and {other_paper} are papers outside the answer "
        "(in list questions, mostly the near-misses an answer cites to rule them out)",
        f"- gold evidence entries the answers cited: {share(gold_cited, gold_entries)}",
        f"- citations of a page the agent had been shown: {share(seen, checkable)}",
    ]
    if unseen:
        lines.append(
            f"- never shown, so cited from memory: {', '.join(unseen)} "
            f"({unseen_gold} of them on a gold page)"
        )
    if no_citations:
        lines.append(f"- answers with gold evidence but no citations: {', '.join(no_citations)}")
    lines += [
        "",
        "Citations only cover what an answer cites. A claim made from memory without a "
        "citation (as in q020, where the agent wrote that it relied on what it knew of six "
        "papers) is not caught by this check.",
    ]
    return lines


def main_stats(args: Any) -> int:
    import sys

    try:
        text = render(args.run_dir, args.question_file, args.processed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    (args.run_dir / "stats.md").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0
