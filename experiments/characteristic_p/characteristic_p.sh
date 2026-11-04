#!/usr/bin/env bash
# Characteristic p of the QELM against the space-time volume N T (issue #285):
# experiments/nonlinear_M/nonlinear_M.py run on a grid of encodings, sizes N,
# depths T and seeds (one fixed Haar U(4) circuit per seed), with a p grid
# that is dense in the expected number of measurements p N T.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/characteristic_p/characteristic_p.sh
#
# Each run writes metrics.json to
#     $OUT_DIR/<encoding>/N<N>_T<T>/seed<seed>/
# and experiments/characteristic_p/plot_characteristic_p.py reads that tree.
# Environment variables change the defaults (lists are space-separated):
#     OUT_ROOT=/scratch/qelm     write under /scratch/qelm instead of outputs
#     OUT_DIR=...                write the sweep exactly there
#     ENCODINGS="ghz bell"       encodings (default "ghz w bell charge")
#     SIZES="4:2,4,8 6:3,6"      N:T1,T2,... pairs (default below)
#     SEEDS="$(seq 42 51)"       seeds, one circuit each (default 42..47)
#     PER_P=2000                 trajectories per p (default 1000)
#     MAX_JOBS=4                 runs at once (default 1)
#     SKIP_DONE=1                skip the runs that already have a metrics.json
#
# Cost: one trajectory costs about 8 ms at N = 8, T = 16 and scales like
# 2^N T. The default grid (N = 10 included) is about 15 CPU hours with
# PER_P=1000 and 6 seeds; on the cluster, use MAX_JOBS to fill the node.
#
# A failed run does not stop the others: the sweep lists the failed runs in
# $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

EXPERIMENT="characteristic_p"
read -r -a ENCODINGS <<< "${ENCODINGS:-ghz w bell charge}"
read -r -a SIZES <<< "${SIZES:-4:2,4,8,16 6:3,6,12,24 8:4,8,16,32 10:5,10,20}"
read -r -a SEEDS <<< "${SEEDS:-$(seq -s ' ' 42 47)}"
PER_P="${PER_P:-1000}"
MAX_JOBS="${MAX_JOBS:-1}"
SKIP_DONE="${SKIP_DONE:-0}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# Largest runs first, so that MAX_JOBS stays busy at the end of the queue.
RUNS=()
for (( i=${#SIZES[@]}-1; i>=0; i-- )); do
  N="${SIZES[$i]%%:*}"
  IFS=',' read -r -a TS <<< "${SIZES[$i]#*:}"
  for (( j=${#TS[@]}-1; j>=0; j-- )); do
    for ENC in "${ENCODINGS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        RUNS+=("$ENC $N ${TS[$j]} $SEED")
      done
    done
  done
done

FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "encoding N T seed".
run_one() {
  set -uo pipefail
  local ENC N T SEED RUN_DIR P_LIST status
  read -r ENC N T SEED <<< "$1"
  RUN_DIR="$OUT_DIR/$ENC/N${N}_T${T}/seed${SEED}"
  if [[ "$SKIP_DONE" == "1" && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- skip $1 (done)"
    return 0
  fi
  mkdir -p "$(dirname "$RUN_DIR")"
  P_LIST="$(uv run --quiet python experiments/characteristic_p/characteristic_p.py grid "$N" "$T")"
  echo "--- encoding=$ENC N=$N T=$T seed=$SEED samples_per_p=$PER_P"
  uv run python experiments/nonlinear_M/nonlinear_M.py \
    "N=$N" \
    "seed=$SEED" \
    "nonlinear_task.encoding=$ENC" \
    "nonlinear_task.depth_rule=$T" \
    "nonlinear_task.samples_per_p=$PER_P" \
    "nonlinear_task.p_values=$P_LIST" \
    "nonlinear_task.modes=[raw]" \
    "nonlinear_task.mlp_reference=false" \
    "nonlinear_task.quantum_baseline=true" \
    "nonlinear_task.save_arrays=false" \
    "hydra.run.dir=$RUN_DIR" > /dev/null 2>"$RUN_DIR.log"
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): $1, see $RUN_DIR.log" >&2
    return 1
  fi
  rm -f "$RUN_DIR.log"
}
export -f run_one
export OUT_DIR PER_P FAILED_LOG SKIP_DONE

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
