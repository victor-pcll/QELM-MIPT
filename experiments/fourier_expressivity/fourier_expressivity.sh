#!/usr/bin/env bash
# Fourier expressivity of the monitored QELM (issue #309):
# experiments/fourier_expressivity/fourier_expressivity.py on a grid of
# dynamics, sizes N and seeds (one fixed circuit per seed), depth T = N, with
# every p of P_VALUES in each run.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/fourier_expressivity/fourier_expressivity.sh
#
# Each run writes metrics.json to $OUT_DIR/<model>_N<N>_T<T>_seed<seed>/.
# Environment variables change the defaults (lists are space-separated):
#     OUT_ROOT=/scratch/qelm      write under /scratch/qelm instead of outputs
#     OUT_DIR=...                 write the sweep exactly there
#     MODELS="haar u1 z2 clifford"
#     SIZES="4:10 6:10 8:10 10:2" N:number-of-seeds pairs (seeds 42, 43, ...)
#     T_FACTOR=1                  depth T = T_FACTOR * N
#     P_VALUES="0 0.05 ... 1"     measurement probabilities (default 21 values)
#     MAX_JOBS=4                  runs at once, one BLAS thread each
#     SKIP_DONE=1                 skip the runs that already have a metrics.json
#
# Cost per run (21 values of p, T = N, one core): N = 6 about 1 s, N = 8
# about 1 min, N = 10 about 40 min (the Heisenberg evolution of 2^N x 2^N
# operators). The default grid takes about 1.5 hours with MAX_JOBS=4.
#
# A failed run does not stop the others: the sweep lists the failed runs in
# $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

EXPERIMENT="fourier_expressivity"
read -r -a MODELS <<< "${MODELS:-haar u1 z2 clifford}"
read -r -a SIZES <<< "${SIZES:-4:10 6:10 8:10 10:2}"
T_FACTOR="${T_FACTOR:-1}"
P_VALUES="${P_VALUES:-0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1}"
MAX_JOBS="${MAX_JOBS:-4}"
SKIP_DONE="${SKIP_DONE:-0}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
if ! [[ "$T_FACTOR" =~ ^[1-9][0-9]*$ ]]; then
  echo "T_FACTOR must be a positive integer, got '$T_FACTOR'" >&2
  exit 2
fi
for SIZE in "${SIZES[@]}"; do
  if ! [[ "$SIZE" =~ ^[0-9]+:[1-9][0-9]*$ ]]; then
    echo "SIZES must hold N:seeds pairs, got '$SIZE'" >&2
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
  NSEEDS="${SIZES[$i]#*:}"
  for MODEL in "${MODELS[@]}"; do
    for (( s=0; s<NSEEDS; s++ )); do
      RUNS+=("$MODEL $N $(( T_FACTOR * N )) $(( 42 + s ))")
    done
  done
done

FAILED_LOG="$OUT_DIR/failed_runs.txt"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "model N T seed".
run_one() {
  set -uo pipefail
  local MODEL N T SEED RUN_DIR status
  read -r MODEL N T SEED <<< "$1"
  RUN_DIR="$OUT_DIR/${MODEL}_N${N}_T${T}_seed${SEED}"
  if [[ "$SKIP_DONE" == "1" && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- model=$MODEL N=$N T=$T seed=$SEED already done, skipped"
    return 0
  fi
  echo "--- model=$MODEL N=$N T=$T seed=$SEED"
  mkdir -p "$RUN_DIR"
  # shellcheck disable=SC2086
  uv run python experiments/fourier_expressivity/fourier_expressivity.py \
    --model "$MODEL" --N "$N" --T "$T" --seed "$SEED" \
    --p $P_VALUES --out "$RUN_DIR" > "$RUN_DIR.log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): $1, see $RUN_DIR.log" >&2
    return 1
  fi
  rm -f "$RUN_DIR.log"
}
export -f run_one
export OUT_DIR P_VALUES FAILED_LOG SKIP_DONE
if (( MAX_JOBS > 1 )); then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
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
