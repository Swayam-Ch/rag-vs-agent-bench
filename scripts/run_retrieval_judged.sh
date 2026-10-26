#!/usr/bin/env bash
# Retrieval with judged answers on the current question set: the strongest sweep setup
# (hybrid, bge-base embeddings, cross-encoder reranker) and BM25 again for comparison.
# Each run lands in results/<UTC time>/; this script appends "<setup> <repeat> <run dir>"
# to results/retrieval-judged-runs.txt.
#
#   export ANTHROPIC_API_KEY=...
#   scripts/run_retrieval_judged.sh        # about $0.90
#
# SETUPS lists "<setup>:<runs>"; bm25 goes through `ragbench run`, anything else through
# scripts/run_setup_judged.py. A run starts only if what this script has spent so far plus
# a conservative estimate stays under BUDGET (USD, default 2.00).
set -euo pipefail
cd "$(dirname "$0")/.."

SETUPS=${SETUPS:-"hybrid-bge+rerank:1 bm25:1"}
BUDGET=${BUDGET:-2.00}
LIST=results/retrieval-judged-runs.txt
ESTIMATE=0.60  # one judged run of a retrieval method; the committed bm25 run cost $0.42

spent() {
    .venv/bin/python - "$LIST" <<'EOF'
import json, sys
from pathlib import Path
total = 0.0
list_file = Path(sys.argv[1])
for line in list_file.read_text().splitlines() if list_file.exists() else []:
    if not line.strip():
        continue
    for rows in Path(line.split()[2]).glob("*.jsonl"):
        for raw in rows.read_text().splitlines():
            r = json.loads(raw)
            # an agent row repeats its cost under "answer"; count it once
            writer = "agent" if r.get("agent") else "answer"
            for part in (writer, "judge"):
                total += (r.get(part) or {}).get("cost_usd") or 0.0
print(f"{total:.2f}")
EOF
}

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set. Export it in this shell first." >&2
    exit 1
fi

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
for entry in $SETUPS; do
    n=${entry##*:}
    (( n > max_runs )) && max_runs=$n
done

for rep in $(seq 1 "$max_runs"); do
    for entry in $SETUPS; do
        setup=${entry%%:*}
        runs=${entry##*:}
        (( rep > runs )) && continue
        so_far=$(spent)
        if awk -v s="$so_far" -v n="$ESTIMATE" -v b="$BUDGET" 'BEGIN { exit !(s + n > b) }'; then
            echo "Stopping: \$$so_far spent, the next $setup run (about \$$ESTIMATE) would pass the \$$BUDGET cap."
            exit 0
        fi
        echo "== $setup, run $rep of $runs (spent so far: \$$so_far)"
        if [[ "$setup" == "bm25" ]]; then
            cmd=(.venv/bin/ragbench run --methods bm25 --judge)
        else
            cmd=(.venv/bin/python scripts/run_setup_judged.py "$setup")
        fi
        # both print "run <folder>: ..." first; keep their output visible too
        log=$("${cmd[@]}" | tee /dev/stderr)
        run_dir=$(printf '%s\n' "$log" | sed -n 's/^run \([^:]*\):.*/\1/p' | head -1)
        if [[ -z "$run_dir" ]]; then
            echo "could not find the run folder in the output" >&2
            exit 1
        fi
        echo "$setup $rep results/$run_dir" >> "$LIST"
    done
done

echo "Done. Spent: \$$(spent). Compare with: .venv/bin/python scripts/compare_retrieval.py"
