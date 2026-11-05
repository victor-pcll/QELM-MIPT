#!/usr/bin/env bash
# Learn S_R and the purity of the reference qubit from the final state, on
# Clifford circuits (issue #292): runs
# experiments/clifford_state_readout/clifford_state_readout.py for every N and
# seed, then draws the figures with plot_clifford_state_readout.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/clifford_state_readout/clifford_state_readout.sh
#
# Each run writes metrics.json, its log and its Hydra config to
#     outputs/clifford_state_readout/<date_time>/N<N>_seed<seed>/
# and the figures and summary.json land in the sweep folder.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there
#     N_VALUES="6 8 10 12 16 20"  system qubits, even
#     SEEDS="42 43 44"            one run per seed
#     PER_P=2000                  trajectories per p (task.samples_per_p)
#     SCRAMBLE=0                  steps without measurement (task.scramble_steps);
#                                 2N for twice the size (Gullans & Huse 2020)
#     READOUTS="[chance,...]"     readouts (task.readouts), Hydra list
#     SAVE_ARRAYS=false           also write trajectories.npz
#     SKIP_DONE=true              skip the runs whose metrics.json exists
#     MAX_JOBS=1                  runs at once

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

EXPERIMENT="clifford_state_readout"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10 12 16 20}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
PER_P="${PER_P:-2000}"
SCRAMBLE="${SCRAMBLE:-0}"
READOUTS="${READOUTS:-}"
SAVE_ARRAYS="${SAVE_ARRAYS:-false}"
SKIP_DONE="${SKIP_DONE:-true}"
MAX_JOBS="${MAX_JOBS:-1}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got $MAX_JOBS." >&2
  exit 1
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

run_one() {
  local N="$1" SEED="$2" RUN_DIR SCR="$SCRAMBLE"
  if [[ "$SCR" == 2N ]]; then
    SCR=$((2 * N))
  fi
  RUN_DIR="$OUT_DIR/N${N}_seed${SEED}"
  if [[ "$SKIP_DONE" == true && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- skip N=$N seed=$SEED (done)"
    return 0
  fi
  local ARGS=(
    "N=$N"
    "seed=$SEED"
    "task.samples_per_p=$PER_P"
    "task.scramble_steps=$SCR"
    "task.save_arrays=$SAVE_ARRAYS"
    "hydra.run.dir=$RUN_DIR"
  )
  if [[ -n "$READOUTS" ]]; then
    ARGS+=("task.readouts=$READOUTS")
  fi
  mkdir -p "$RUN_DIR"
  uv run python experiments/clifford_state_readout/clifford_state_readout.py \
    "${ARGS[@]}" > "$RUN_DIR/run.log" 2>&1 \
    || { echo "FAILED: N=$N seed=$SEED, see $RUN_DIR/run.log" >&2; return 1; }
  echo "--- done N=$N seed=$SEED"
}

# Largest sizes first, so that the parallel runs end together.
SORTED_N=($(printf '%s\n' "${N_VALUES[@]}" | sort -rn))
PIDS=()
for N in "${SORTED_N[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
      sleep 0.2
    done
    run_one "$N" "$SEED" &
    PIDS+=("$!")
  done
done
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
if (( STATUS )); then
  echo "Some runs failed, see above." >&2
  exit 1
fi

uv run python experiments/clifford_state_readout/plot_clifford_state_readout.py "$OUT_DIR"
echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
