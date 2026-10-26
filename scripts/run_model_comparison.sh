#!/usr/bin/env bash
# The agent on all 20 questions with two models, several runs each, every answer judged.
# Each run lands in results/<UTC time>/ like any `ragbench run`; this script also appends
# "<model> <repeat> <run dir>" to results/model-comparison-runs.txt, which
# scripts/compare_models.py reads.
#
#   export ANTHROPIC_API_KEY=...        # and ANTHROPIC_WORKSPACE_ID if your key needs it
#   scripts/run_model_comparison.sh --smoke     # one question per model, a few cents
#   scripts/run_model_comparison.sh             # the comparison, about $3
#
# MODELS lists "<model>:<runs>". Runs are interleaved (round 1 of every model, then
# round 2, ...), so stopping early still leaves every model compared.
# A spending cap (BUDGET, in USD, default 4.75) covers everything this script has run so
# far, smoke test included: a run starts only if what was spent plus a conservative
# estimate for that run stays under the cap.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"claude-haiku-5-5:3 claude-sonnet-5-5:2"}
BUDGET=${BUDGET:-4.75}
LIST=results/model-comparison-runs.txt
SMOKE_LIST=results/model-comparison-smoke.txt
QUESTIONS=()
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
    LIST=$SMOKE_LIST
    QUESTIONS=(--questions q001)
fi

# Cost of one full 20-question run incl. judging, with headroom over the committed
# run's token counts (sonnet $0.91 + $0.20 judging).
estimate() {
    if [[ $SMOKE == 1 ]]; then echo 0.15; return; fi
    case "$1" in
        claude-haiku-*) echo 0.40 ;;
        claude-sonnet-*) echo 1.40 ;;
        claude-opus-*) echo 2.50 ;;
        *) echo 2.50 ;;
    esac
}

# Agent plus judging cost of every run listed so far (both list files).
spent() {
    .venv/bin/python - "$LIST" "$SMOKE_LIST" <<'EOF'
import json, sys
from pathlib import Path
total = 0.0
for list_file in map(Path, sys.argv[1:]):
    if not list_file.exists():
        continue
    for line in list_file.read_text().splitlines():
        if not line.strip():
            continue
        rows = Path(line.split()[2]) / "agent.jsonl"
        for raw in rows.read_text().splitlines() if rows.exists() else []:
            r = json.loads(raw)
            total += (r.get("agent") or {}).get("cost_usd") or 0.0
            total += (r.get("judge") or {}).get("cost_usd") or 0.0
print(f"{total:.2f}")
EOF
}

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set. Export it in this shell first." >&2
    exit 1
fi

# A free call that fails fast on a dead or wrong key, before any run starts.
.venv/bin/python - <<'EOF'
import sys
import anthropic
try:
    anthropic.Anthropic().models.list(limit=1)
except anthropic.AuthenticationError:
    sys.exit("the API key was rejected (inactive or wrong key)")
except anthropic.PermissionDeniedError:
    sys.exit("the API key has no access (check ANTHROPIC_WORKSPACE_ID)")
print("API key OK")
EOF

max_runs=1
for entry in $MODELS; do
    n=${entry##*:}
    (( n > max_runs )) && max_runs=$n
done
[[ $SMOKE == 1 ]] && max_runs=1

for rep in $(seq 1 "$max_runs"); do
    for entry in $MODELS; do
        model=${entry%%:*}
        runs=${entry##*:}
        (( rep > runs )) && continue
        so_far=$(spent)
        next=$(estimate "$model")
        if awk -v s="$so_far" -v n="$next" -v b="$BUDGET" 'BEGIN { exit !(s + n > b) }'; then
            echo "Stopping: \$$so_far spent, the next $model run (about \$$next) would pass the \$$BUDGET cap."
            echo "Compare what ran with: .venv/bin/python scripts/compare_models.py $LIST"
            exit 0
        fi
        echo "== $model, run $rep of $runs (spent so far: \$$so_far)"
        # `ragbench run` prints "run <folder>: ..." first; keep its output visible too
        log=$(.venv/bin/ragbench run --methods agent --model "$model" --judge \
            ${QUESTIONS[@]+"${QUESTIONS[@]}"} | tee /dev/stderr)
        run_dir=$(printf '%s\n' "$log" | sed -n 's/^run \([^:]*\):.*/\1/p' | head -1)
        if [[ -z "$run_dir" ]]; then
            echo "could not find the run folder in the output" >&2
            exit 1
        fi
        echo "$model $rep results/$run_dir" >> "$LIST"
    done
done

echo "Done. Total spent: \$$(spent). Compare with: .venv/bin/python scripts/compare_models.py $LIST"
