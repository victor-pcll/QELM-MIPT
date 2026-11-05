#!/usr/bin/env bash
# Early-layers hypothesis of the linear QELM on M
# (experiments/early_layers/early_layers.py, issue #288), swept over the
# configurations (gates and encoding) and the seed: one fixed circuit per seed.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/early_layers/early_layers.sh
#
# Each run writes metrics.json to
#     outputs/early_layers/<date_time>/<gates>_<encoding>_N<N>_seed<seed>/
# which git ignores; experiments/early_layers/plot_early_layers.py draws the
# figures from that folder. Environment variables change the defaults
# (lists are space-separated):
#     OUT_ROOT=/scratch/qelm     write under /scratch/qelm instead of outputs
#     OUT_DIR=...                write the sweep exactly there
#     N_VALUES="8 10"            numbers of system qubits (default "8")
#     SEEDS="$(seq 42 61)"       seeds, one circuit each (default 42..51)
#     CONFIGS="haar:bell u1:bell" gates:encoding pairs (default
#                                "haar:bell haar:ghz haar:charge u1:bell u1:charge")
#                                gates: haar, u1, z2, dipole, diagonal
#     BACKGROUND=01100110        early_task.background: the Bell encoding on that
#                                background (default: none, |0...0>)
#     PER_P=8000                 trajectories per p (default 4000)
#     DEPTH="2N"                 early_task.depth_rule (default 2N)
#     P_VALUES="0.1 0.2"         overrides early_task.p_values
#     MAX_JOBS=4                 runs at once (default 1)
#     SKIP_DONE=1                skip the runs that already have a metrics.json
#
# Cost: N = 8, T = 16, 8 values of p and 4000 trajectories per p take about
# 10 minutes per run on one core.
#
# A failed run does not stop the others: the sweep goes to the end, lists the
# failed runs in $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV
# One thread per run: MAX_JOBS sets the parallelism.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

EXPERIMENT="early_layers"
read -r -a N_VALUES <<< "${N_VALUES:-8}"
read -r -a SEEDS <<< "${SEEDS:-$(seq -s ' ' 42 51)}"
read -r -a CONFIGS <<< "${CONFIGS:-haar:bell haar:ghz haar:charge u1:bell u1:charge}"
PER_P="${PER_P:-4000}"
BACKGROUND="${BACKGROUND:-}"
DEPTH="${DEPTH:-2N}"
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

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

RUNS=()
for CONFIG in "${CONFIGS[@]}"; do
  for N in "${N_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUNS+=("$CONFIG $N $SEED")
    done
  done
done

FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "gates:encoding N seed".
run_one() {
  set -uo pipefail
  local CONFIG N SEED GATES ENCODING RUN_DIR status
  read -r CONFIG N SEED <<< "$1"
  GATES="${CONFIG%%:*}"
  ENCODING="${CONFIG##*:}"
  RUN_DIR="$OUT_DIR/${GATES}_${ENCODING}_N${N}_seed${SEED}"
  if [[ "$SKIP_DONE" == "1" && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- skip $CONFIG N=$N seed=$SEED (done)"
    return 0
  fi
  echo "--- $CONFIG N=$N seed=$SEED samples_per_p=$PER_P depth=$DEPTH"
  uv run python experiments/early_layers/early_layers.py \
    "N=$N" \
    "seed=$SEED" \
    "early_task.gates=$GATES" \
    "early_task.encoding=$ENCODING" \
    "early_task.samples_per_p=$PER_P" \
    "early_task.depth_rule=$DEPTH" \
    ${P_LIST:+"early_task.p_values=$P_LIST"} \
    ${BACKGROUND:+"early_task.background=\"$BACKGROUND\""} \
    "hydra.run.dir=$RUN_DIR" > "$RUN_DIR.log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): $CONFIG N=$N seed=$SEED" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR PER_P DEPTH P_LIST BACKGROUND FAILED_LOG SKIP_DONE

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

echo "Done: ${#RUNS[@]} runs in $OUT_DIR"
