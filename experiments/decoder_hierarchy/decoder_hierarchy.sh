#!/usr/bin/env bash
# Three decoders of the charge on the same monitored U(1) trajectories
# (issue #307): runs experiments/decoder_hierarchy/decoder_hierarchy.py for
# every N and seed, then draws the figures with plot_decoder_hierarchy.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/decoder_hierarchy/decoder_hierarchy.sh
#
# Each run writes metrics.json and its Hydra config to
#     outputs/decoder_hierarchy/<date_time>/N<N>_seed<seed>/
# and the figures and summary.json land in the sweep folder.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there
#     N_VALUES="6 8 10 12"        chain sizes (dense state vectors: 2**N)
#     SEEDS="42 43 44"            one run per seed
#     PER_P=4000                  trajectories per p (task.samples_per_p)
#     P_VALUES="[0.0,0.1]"        measurement probabilities (task.p_values)
#     SKIP_DONE=true              skip the runs whose metrics.json exists
#     MAX_JOBS=1                  runs at once

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

EXPERIMENT="decoder_hierarchy"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10 12}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
PER_P="${PER_P:-4000}"
P_VALUES="${P_VALUES:-}"
SKIP_DONE="${SKIP_DONE:-true}"
MAX_JOBS="${MAX_JOBS:-1}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'." >&2
    exit 1
fi
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$OUT_DIR"

run_one() {
    local N="$1" seed="$2"
    local dir="$OUT_DIR/N${N}_seed${seed}"
    if [[ "$SKIP_DONE" == "true" && -f "$dir/metrics.json" ]]; then
        echo "Skip N=$N seed=$seed (done)"
        return
    fi
    local extra=()
    [[ -n "$P_VALUES" ]] && extra+=("task.p_values=$P_VALUES")
    uv run --quiet python "experiments/$EXPERIMENT/$EXPERIMENT.py" \
        "N=$N" "seed=$seed" "task.samples_per_p=$PER_P" "${extra[@]}" \
        "hydra.run.dir=$dir" > "$OUT_DIR/N${N}_seed${seed}.log" 2>&1
    echo "Done N=$N seed=$seed"
}

# Largest sizes first, so that the longest runs start early.
for (( k=${#N_VALUES[@]}-1; k>=0; k-- )); do
    for seed in "${SEEDS[@]}"; do
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do wait -n; done
        run_one "${N_VALUES[$k]}" "$seed" &
    done
done
wait

uv run --quiet python "experiments/$EXPERIMENT/plot_$EXPERIMENT.py" "$OUT_DIR"
echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
