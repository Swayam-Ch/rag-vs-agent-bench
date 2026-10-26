"""One chart for the README: accuracy by question type, one bar per method.

    ragbench chart results/2026-10-26T120551Z     # -> results/<run>/accuracy_by_type.png

Grouped bars, because the story is a comparison within each question type (which method
holds up on negation questions?) and the types are categories, not a time series.
Numbers come from the per-question rows, through the same summary code as the tables.

Design notes (validated palette, checked with a colorblindness validator):
* methods keep fixed colours in a fixed order, whatever subset a run has;
* thin bars with a gap between neighbours, recessive grid, no chart junk;
* a legend names every method, so colour is never the only cue; the README shows
  the same numbers as a table next to the chart (two of the colours are light).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragbench.eval.questions import TYPES
from ragbench.eval.run import summarize as summarize_rows
from ragbench.eval.summarize import load_run

# categorical slots 1-5 of the validated palette, light mode; a method always gets
# the same colour, so charts of different runs can be compared at a glance
METHOD_COLORS = {
    "bm25": "#2a78d6",
    "vector": "#eb6834",
    "hybrid": "#1baf7a",
    "agent": "#eda100",
    "keyword": "#e87ba4",
}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
FILENAME = "accuracy_by_type.png"


def accuracy_by_type(run_dir: Path) -> dict[str, Any]:
    """{"types": [(type, n_questions)], "methods": {method: [accuracy or None per type]}}."""
    _, rows = load_run(run_dir)
    summaries = {m: summarize_rows(r) for m, r in rows.items()}
    if not any("answers" in s for s in summaries.values()):
        raise ValueError(f"{run_dir} has no judged answers; run it with --judge")
    first = next(iter(rows.values()))
    counts = {t: sum(r["type"] == t for r in first) for t in TYPES}
    types = [(t, counts[t]) for t in TYPES if counts[t]]
    methods = {
        m: [s.get("answers", {}).get("by_type", {}).get(t) for t, _ in types]
        for m, s in summaries.items()
        if "answers" in s
    }
    return {"types": types, "methods": methods}


def plot(run_dir: Path, out: Path | None = None) -> Path:
    """Draw the chart and save it as a PNG (light background, 2x resolution)."""
    import matplotlib

    matplotlib.use("Agg")  # no window; works in CI and over ssh
    import matplotlib.pyplot as plt

    data = accuracy_by_type(run_dir)
    types, methods = data["types"], data["methods"]
    n_methods = len(methods)
    group = 0.78  # share of each type's slot taken by its bars
    width = group / n_methods

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (method, values) in enumerate(methods.items()):
        xs = [t + (i - (n_methods - 1) / 2) * width for t in range(len(types))]
        heights = [0.0 if v is None else v for v in values]
        ax.bar(
            xs,
            heights,
            width=width,
            color=METHOD_COLORS.get(method, "#8a8984"),
            edgecolor=SURFACE,  # a thin surface-coloured gap between neighbouring bars
            linewidth=1.6,
            label=method,
            zorder=3,
        )

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(
        [f"{t.replace('_', ' ')}\n{n} question{'s' if n != 1 else ''}" for t, n in types]
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.axhline(0, color=TEXT_2, linewidth=0.9, zorder=4)  # baseline drawn over the bar gaps
    ax.set_ylabel("accuracy", color=TEXT_2, fontsize=9)

    n_questions = sum(n for _, n in types)
    fig.text(0.06, 0.95, "Accuracy by question type", fontsize=13, color=TEXT, weight="bold")
    fig.text(
        0.06,
        0.895,
        f"{n_questions} questions, answers graded by an LLM judge (list questions: F1 of the "
        f"papers named) · run {run_dir.name}",
        fontsize=8.5,
        color=TEXT_2,
    )
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=n_methods,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)  # text wears text colours; the swatch carries identity
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.16)

    out = out or run_dir / FILENAME
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
