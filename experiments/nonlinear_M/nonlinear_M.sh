#!/usr/bin/env bash
# Non-linear features of the record M (experiments/nonlinear_M/nonlinear_M.py,
# issue #269), swept over N and the seed: one fixed Haar U(4) circuit per seed,
# the reference qubit in a Bell pair with the middle site.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/nonlinear_M/nonlinear_M.sh
#
# Each run writes metrics.json (and the arrays, unless SAVE_ARRAYS=false) to
#     outputs/nonlinear_M/<date_time>/N<N>_seed<seed>/
# which git ignores; experiments/nonlinear_M/plot_nonlinear_M.py draws the
# figures from that folder. Environment variables change the defaults
# (lists are space-separated):
#     OUT_ROOT=/scratch/qelm     write under /scratch/qelm instead of outputs
#     OUT_DIR=...                write the sweep exactly there
#     N_VALUES="6 8 10"          numbers of system qubits (default "6 8")
#     SEEDS="$(seq 42 61)"       seeds, one circuit each (default 42..61, N_c = 20)
#     PER_P=8000                 trajectories per p (default 2000)
#     DEPTH="2N"                 nonlinear_task.depth_rule (default N)
#     P_VALUES="0 0.1 0.2"       overrides nonlinear_task.p_values
#     MODES="raw xor_local"      overrides nonlinear_task.modes
#     MLP=false                  skip the MLP reference
#     FIXED=false                new gates per trajectory (the null control;
#                                default true, one fixed circuit per seed)
#     SAVE_ARRAYS=false          skip the .npz (large at N = 12)
#     MAX_JOBS=4                 runs at once (default 1)
#     SKIP_DONE=1                skip the runs that already have a metrics.json
#
# Cluster sizes: N = 10 costs about 4x N = 8 per trajectory and N = 12 about
# 16x; with PER_P=8000 and the 19 default p, plan a few hours per run at
# N = 12 and use MAX_JOBS to fill the node.
#
# A failed run does not stop the others: the sweep goes to the end, lists the
# failed runs in $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

EXPERIMENT="nonlinear_M"
read -r -a N_VALUES <<< "${N_VALUES:-6 8}"
read -r -a SEEDS <<< "${SEEDS:-$(seq -s ' ' 42 61)}"
PER_P="${PER_P:-2000}"
DEPTH="${DEPTH:-N}"
MLP="${MLP:-true}"
FIXED="${FIXED:-true}"
SAVE_ARRAYS="${SAVE_ARRAYS:-true}"
MAX_JOBS="${MAX_JOBS:-1}"
SKIP_DONE="${SKIP_DONE:-0}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
P_LIST=""
if [[ -n "${P_VALUES:-}" ]]; then
  P_LIST="[$(echo "$P_VALUES" | xargs | tr ' ' ',')]"
fi
M_LIST=""
if [[ -n "${MODES:-}" ]]; then
  M_LIST="[$(echo "$MODES" | xargs | tr ' ' ',')]"
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

RUNS=()
for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUNS+=("$N $SEED")
  done
done

FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "N seed".
run_one() {
  set -uo pipefail
  local N SEED RUN_DIR status
  read -r N SEED <<< "$1"
  RUN_DIR="$OUT_DIR/N${N}_seed${SEED}"
  if [[ "$SKIP_DONE" == "1" && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- skip N=$N seed=$SEED (done)"
    return 0
  fi
  echo "--- N=$N seed=$SEED samples_per_p=$PER_P depth=$DEPTH fixed=$FIXED"
  uv run python experiments/nonlinear_M/nonlinear_M.py \
    "N=$N" \
    "seed=$SEED" \
    "nonlinear_task.samples_per_p=$PER_P" \
    "nonlinear_task.depth_rule=$DEPTH" \
    "nonlinear_task.mlp_reference=$MLP" \
    "haar_circuit.fixed_instance=$FIXED" \
    "nonlinear_task.save_arrays=$SAVE_ARRAYS" \
    ${P_LIST:+"nonlinear_task.p_values=$P_LIST"} \
    ${M_LIST:+"nonlinear_task.modes=$M_LIST"} \
    "hydra.run.dir=$RUN_DIR"
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): N=$N seed=$SEED" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR PER_P DEPTH MLP FIXED SAVE_ARRAYS P_LIST M_LIST FAILED_LOG SKIP_DONE

QUEUE_STATUS=0
printf '%s\0' "${RUNS[@]}" \
  | xargs -0 -P "$MAX_JOBS" -I {} bash -c 'run_one "$1"' _ {} \
  || QUEUE_STATUS=$?

if [[ -s "$FAILED_LOG" ]]; then
  echo "$(wc -l < "$FAILED_LOG" | tr -d ' ') of ${#RUNS[@]} runs failed, listed in $FAILED_LOG:" >&2
  cat "$FAILED_LOG" >&2
  exit 1
fi
rm -f "$FAILED_LOG"
if (( QUEUE_STATUS != 0 )); then
  echo "xargs stopped with exit $QUEUE_STATUS" >&2
  exit "$QUEUE_STATUS"
fi

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
