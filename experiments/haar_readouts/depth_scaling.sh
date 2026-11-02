#!/usr/bin/env bash
# Depth T of the monitored Haar circuit read by the QELM (issue #268): the
# charge task of experiments/haar_readouts/haar_readouts.py on one fixed
# circuit per seed, swept over N, the depth rule T(N) and the seed.
#
# Three depth rules (readouts_task.depth_rule):
#     10   the same T = 10 rounds for every N (Dehghani et al. 2023)
#     N    T = N, the light cone of the brickwork covers the chain
#     2N   T = 2N, the stationary regime of the MIPT
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/haar_readouts/depth_scaling.sh
#
# Each run writes its arrays, metrics.json (with the timings and the
# condition numbers) and Hydra config to
#     outputs/haar_depth/<date_time>/haar_fixed_N<N>_T<rule>_seed<seed>/
# which git ignores; experiments/haar_readouts/plot_depth_scaling.py draws the
# figures from that folder. Environment variables change the defaults (lists
# are space-separated):
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=...                 write the sweep exactly there
#     N_VALUES="6 8 10 12"        numbers of system qubits
#     DEPTHS="10 N 2N"            depth rules
#     SEEDS="42 ... 51"           seeds, one fixed Haar circuit each
#     PER_P=2000                  trajectories per p (readouts_task.samples_per_p)
#     P_VALUES="0 0.05 ..."       measurement probabilities (default below)
#     READOUTS="ridge"            readouts (readouts_task.readouts)
#     FEATURES="all"              feature sets (readouts_task.features)
#     MAX_JOBS=4                  runs at once (default 1); each run then
#                                 uses one BLAS thread
#
# A failed run does not stop the others: the sweep goes to the end, lists the
# failed runs in $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="haar_depth"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10 12}"
read -r -a DEPTHS <<< "${DEPTHS:-10 N 2N}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
PER_P="${PER_P:-2000}"
MAX_JOBS="${MAX_JOBS:-1}"
# The grid of the issue, p = 0.05 to 0.4 by 0.025 with 0.14, 0.16, 0.18 and
# 0.22 around p_c, plus p = 0 and 0.0125, 0.025: at T = N and 2N the QELM
# loses the charge before p = 0.05.
P_VALUES="${P_VALUES:-0 0.0125 0.025 0.05 0.075 0.1 0.125 0.14 0.15 0.16 0.175 0.18 0.2 0.22 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4}"
READOUTS="${READOUTS:-ridge}"
FEATURES="${FEATURES:-all}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
for D in "${DEPTHS[@]}"; do
  if ! [[ "$D" =~ ^([1-9][0-9]*|[1-9]?[0-9]*N)$ ]]; then
    echo "DEPTHS must hold numbers or multiples of N (10, N, 2N), got '$D'" >&2
    exit 2
  fi
done
P_LIST="[$(echo "$P_VALUES" | xargs | tr ' ' ',')]"
R_LIST="[$(echo "$READOUTS" | xargs | tr ' ' ',')]"
F_LIST="[$(echo "$FEATURES" | xargs | tr ' ' ',')]"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# Largest runs first, so that the parallel queue ends together.
RUNS=()
for (( i=${#N_VALUES[@]}-1; i>=0; i-- )); do
  for D in "${DEPTHS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUNS+=("${N_VALUES[$i]} $D $SEED")
    done
  done
done

FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "N rule seed".
run_one() {
  set -uo pipefail
  local N D SEED RUN_DIR status
  read -r N D SEED <<< "$1"
  RUN_DIR="$OUT_DIR/haar_fixed_N${N}_T${D}_seed${SEED}"
  echo "--- N=$N depth_rule=$D seed=$SEED samples_per_p=$PER_P"
  uv run python experiments/haar_readouts/haar_readouts.py \
    "models=haar_circuit" \
    "N=$N" \
    "seed=$SEED" \
    "haar_circuit.fixed_instance=true" \
    "readouts_task.depth_rule=$D" \
    "readouts_task.samples_per_p=$PER_P" \
    "readouts_task.p_values=$P_LIST" \
    "readouts_task.readouts=$R_LIST" \
    "readouts_task.features=$F_LIST" \
    "readouts_task.save_arrays=false" \
    "hydra.run.dir=$RUN_DIR" > "$RUN_DIR.log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): N=$N depth_rule=$D seed=$SEED, see $RUN_DIR.log" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR PER_P P_LIST R_LIST F_LIST FAILED_LOG
if (( MAX_JOBS > 1 )); then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fi

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

echo "Done: ${#N_VALUES[@]} x ${#DEPTHS[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
