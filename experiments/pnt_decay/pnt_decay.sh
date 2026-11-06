#!/usr/bin/env bash
# Why the QELM state readout decays with p N T (issue #305):
# experiments/pnt_decay/pnt_decay.py on a grid of sizes N, depths T and seeds
# (one fixed Haar U(4) circuit per seed), with p on a grid in p N T.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/pnt_decay/pnt_decay.sh
#
# Each run writes metrics.json and its Hydra config to
#     $OUT_DIR/<encoding>_N<N>_T<T>_seed<seed>/
# and experiments/pnt_decay/plot_pnt_decay.py reads that folder.
# Environment variables change the defaults (lists are space-separated):
#     OUT_ROOT=/scratch/qelm     write under /scratch/qelm instead of outputs
#     OUT_DIR=...                write the sweep exactly there
#     ENCODINGS="charge bell"    encodings (default charge)
#     SIZES="4:2,4,8 6:3,6"      N:T1,T2,... pairs (default below)
#     SEEDS="$(seq 42 51)"       seeds, one circuit each (default 42..51)
#     PER_P=3000                 trajectories per p (default 3000)
#     PNT_VALUES="0 1 2 4 8"     grid in p N T (default: that of the config)
#     SAMPLE=0                   exact averaged channel only, no trajectories
#                                (signal, V_eff, V_feat: a cheap scan)
#     MAX_JOBS=4                 runs at once (default 1), one BLAS thread each
#     SKIP_DONE=1                skip the runs that already have a metrics.json
#
# Cost per run, about: N = 10, T = 40: 4 min of exact evolution (density
# matrix, 2^N x 2^N) and 15 min of trajectories with PER_P = 3000. N = 12 is
# 16 times the exact part and 4 times the trajectories: run it on the cluster,
# for example SIZES="12:6,12,24,48" MAX_JOBS=16.
#
# A failed run does not stop the others: the sweep lists the failed runs in
# $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

EXPERIMENT="pnt_decay"
read -r -a ENCODINGS <<< "${ENCODINGS:-charge}"
read -r -a SIZES <<< "${SIZES:-4:2,4,8,16 6:3,6,12,24 8:4,8,16,32 10:5,10,20,40}"
read -r -a SEEDS <<< "${SEEDS:-$(seq -s ' ' 42 51)}"
PER_P="${PER_P:-3000}"
MAX_JOBS="${MAX_JOBS:-1}"
SKIP_DONE="${SKIP_DONE:-0}"
SAMPLE="${SAMPLE:-1}"
if [[ "$SAMPLE" != "0" && "$SAMPLE" != "1" ]]; then
  echo "SAMPLE must be 0 or 1, got '$SAMPLE'" >&2
  exit 2
fi
PNT_LIST=""
if [[ -n "${PNT_VALUES:-}" ]]; then
  PNT_LIST="[$(echo "$PNT_VALUES" | xargs | tr ' ' ',')]"
fi
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
for SIZE in "${SIZES[@]}"; do
  if ! [[ "$SIZE" =~ ^[0-9]+:[0-9]+(,[0-9]+)*$ ]]; then
    echo "SIZES must hold N:T1,T2,... pairs, got '$SIZE'" >&2
    exit 2
  fi
done

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
  local ENC N T SEED RUN_DIR status
  read -r ENC N T SEED <<< "$1"
  RUN_DIR="$OUT_DIR/${ENC}_N${N}_T${T}_seed${SEED}"
  if [[ "$SKIP_DONE" == "1" && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- encoding=$ENC N=$N T=$T seed=$SEED already done, skipped"
    return 0
  fi
  echo "--- encoding=$ENC N=$N T=$T seed=$SEED samples_per_p=$PER_P"
  uv run python experiments/pnt_decay/pnt_decay.py \
    "N=$N" \
    "seed=$SEED" \
    "pnt_task.encoding=$ENC" \
    "pnt_task.T=$T" \
    "pnt_task.samples_per_p=$PER_P" \
    "pnt_task.sample=$([[ "$SAMPLE" == "1" ]] && echo true || echo false)" \
    ${PNT_LIST:+"pnt_task.pnt_values=$PNT_LIST"} \
    "hydra.run.dir=$RUN_DIR" > "$RUN_DIR.log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): $1, see $RUN_DIR.log" >&2
    return 1
  fi
  rm -f "$RUN_DIR.log"
}
export -f run_one
export OUT_DIR PER_P PNT_LIST FAILED_LOG SKIP_DONE SAMPLE
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

echo "Done: ${#RUNS[@]} runs in $OUT_DIR"
