#!/usr/bin/env bash
# Which task makes the QELM read the state, which the record M? (issue #311)
#
# Runs experiments/nonlinear_M/nonlinear_M.py on the grid gate family x
# encoding x target x seed: one fixed circuit per seed, the ridge on the
# state (<P_k>, <P_i P_j>), on vec(M), on both, and the Bayes decoder of M.
# experiments/symmetry_regimes/plot_symmetry_regimes.py draws the figures.
#
# Run from anywhere, after `uv sync`:
#
#     MAX_JOBS=4 ./experiments/symmetry_regimes/symmetry_regimes.sh
#
# Each run writes metrics.json to
#     $OUT_DIR/<circuit>_<encoding>_<target>/N<N>_seed<seed>/
# Environment variables change the defaults (lists are space-separated):
#     OUT_ROOT=/scratch/qelm     write under /scratch/qelm instead of outputs
#     OUT_DIR=...                write the sweep exactly there
#     CIRCUITS="haar u1"         gate families (default: haar clifford u1 z2
#                                dipole diagonal)
#     ENCODINGS="bell ghz"       encodings (default: bell charge ghz w)
#     TARGETS="z x"              axes of R (default: z)
#     N=8                        system qubits (default 8)
#     DEPTH=2N                   nonlinear_task.depth_rule (default 2N)
#     SEEDS="$(seq 42 51)"       seeds, one circuit each (default 42..46)
#     PER_P=8000                 trajectories per p (default 3000)
#     P_VALUES="0 0.1 1"         overrides the grid of p below
#     MAX_JOBS=4                 runs at once (default 1)
#     SKIP_DONE=1                skip the runs that already have a metrics.json
#
# A failed run does not stop the others; they are listed in
# $OUT_DIR/failed_runs.txt and the script exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# One thread per run: the parallelism is over runs (MAX_JOBS).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset VIRTUAL_ENV

read -r -a CIRCUITS <<< "${CIRCUITS:-haar clifford u1 z2 dipole diagonal}"
read -r -a ENCODINGS <<< "${ENCODINGS:-bell charge ghz w}"
read -r -a TARGETS <<< "${TARGETS:-z}"
read -r -a SEEDS <<< "${SEEDS:-$(seq -s ' ' 42 46)}"
N="${N:-8}"
DEPTH="${DEPTH:-2N}"
PER_P="${PER_P:-3000}"
# Dense at small p, where the state readout decays (p ~ 4 / (N T)).
P_VALUES="${P_VALUES:-0 0.0125 0.025 0.0375 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0}"
P_LIST="[$(echo "$P_VALUES" | xargs | tr ' ' ',')]"
MAX_JOBS="${MAX_JOBS:-1}"
SKIP_DONE="${SKIP_DONE:-0}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/symmetry_regimes/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

RUNS=()
for SEED in "${SEEDS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    for ENC in "${ENCODINGS[@]}"; do
      for C in "${CIRCUITS[@]}"; do
        RUNS+=("$C $ENC $TARGET $SEED")
      done
    done
  done
done

FAILED_LOG="$OUT_DIR/failed_runs.txt"
: > "$FAILED_LOG"

run_one() {
  set -uo pipefail
  local C ENC TARGET SEED RUN_DIR status
  read -r C ENC TARGET SEED <<< "$1"
  RUN_DIR="$OUT_DIR/${C}_${ENC}_${TARGET}/N${N}_seed${SEED}"
  if [[ "$SKIP_DONE" == "1" && -f "$RUN_DIR/metrics.json" ]]; then
    echo "--- skip $1 (done)"
    return 0
  fi
  mkdir -p "$RUN_DIR"
  echo "--- circuit=$C encoding=$ENC target=$TARGET seed=$SEED"
  uv run python experiments/nonlinear_M/nonlinear_M.py \
    "N=$N" "seed=$SEED" \
    "nonlinear_task.circuit=$C" \
    "nonlinear_task.encoding=$ENC" \
    "nonlinear_task.target=$TARGET" \
    "nonlinear_task.depth_rule=$DEPTH" \
    "nonlinear_task.samples_per_p=$PER_P" \
    "nonlinear_task.p_values=$P_LIST" \
    "nonlinear_task.modes=[raw]" \
    "nonlinear_task.mlp_reference=false" \
    "nonlinear_task.quantum_baseline=true" \
    "nonlinear_task.save_arrays=false" \
    "hydra.run.dir=$RUN_DIR" > "$RUN_DIR.log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): $1, log in $RUN_DIR.log" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR N DEPTH PER_P P_LIST FAILED_LOG SKIP_DONE

QUEUE_STATUS=0
printf '%s\0' "${RUNS[@]}" \
  | xargs -0 -P "$MAX_JOBS" -I {} bash -c 'run_one "$1"' _ {} \
  || QUEUE_STATUS=$?

if [[ -s "$FAILED_LOG" ]]; then
  echo "$(wc -l < "$FAILED_LOG" | tr -d ' ') of ${#RUNS[@]} runs failed:" >&2
  cat "$FAILED_LOG" >&2
  exit 1
fi
rm -f "$FAILED_LOG"
if (( QUEUE_STATUS != 0 )); then
  echo "xargs stopped with exit $QUEUE_STATUS" >&2
  exit "$QUEUE_STATUS"
fi
echo "Done: ${#RUNS[@]} runs in $OUT_DIR"
