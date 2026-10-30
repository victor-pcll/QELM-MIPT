#!/usr/bin/env bash
# Charge of a fully random Haar circuit read by several readouts
# (experiments/haar_readouts/haar_readouts.py), swept over the gate draw
# (new gates per trajectory or one fixed circuit), the circuit (Haar or
# Clifford), N and the seed.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/haar_readouts/haar_readouts.sh
#
# Each run writes its arrays, metrics.json and Hydra config to
#     outputs/haar_readouts/<date_time>/<circuit>_<gates>_N<N>_seed<seed>/
# with <circuit> haar or clifford and <gates> random
# (<model>.fixed_instance=false) or fixed (true),
# which git ignores; experiments/haar_readouts/plot_haar_readouts.py draws
# the figures from that folder. Environment variables change the defaults
# (lists are space-separated):
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=...                 write the sweep exactly there
#     MODELS="haar_circuit clifford_circuit"  circuits (models; default haar_circuit)
#     GATES="random fixed"        gate draws
#     N_VALUES="8"                numbers of system qubits
#     SEEDS="42 43 44"            seeds
#     PER_P=2000                  trajectories per p (readouts_task.samples_per_p)
#     P_VALUES="0 0.1 0.2"        overrides readouts_task.p_values
#     READOUTS="ridge"            overrides readouts_task.readouts
#     FEATURES="X Z ZZ all"       overrides readouts_task.features
#     MAX_JOBS=4                  runs at once (default 1)
#
# A failed run does not stop the others: the sweep goes to the end, lists the
# failed runs in $OUT_DIR/failed_runs.txt and exits with 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="haar_readouts"
read -r -a MODELS <<< "${MODELS:-haar_circuit}"
read -r -a GATES <<< "${GATES:-random fixed}"
read -r -a N_VALUES <<< "${N_VALUES:-8}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
PER_P="${PER_P:-2000}"
MAX_JOBS="${MAX_JOBS:-1}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
for G in "${GATES[@]}"; do
  if [[ "$G" != "random" && "$G" != "fixed" ]]; then
    echo "GATES must hold random and/or fixed, got '$G'" >&2
    exit 2
  fi
done
# P_VALUES, when set, becomes the list readouts_task.p_values=[...].
P_LIST=""
if [[ -n "${P_VALUES:-}" ]]; then
  P_LIST="[$(echo "$P_VALUES" | xargs | tr ' ' ',')]"
fi
R_LIST=""
if [[ -n "${READOUTS:-}" ]]; then
  R_LIST="[$(echo "$READOUTS" | xargs | tr ' ' ',')]"
fi
F_LIST=""
if [[ -n "${FEATURES:-}" ]]; then
  F_LIST="[$(echo "$FEATURES" | xargs | tr ' ' ',')]"
fi
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

for MODEL in "${MODELS[@]}"; do
  if [[ "$MODEL" != "haar_circuit" && "$MODEL" != "clifford_circuit" ]]; then
    echo "MODELS must hold haar_circuit and/or clifford_circuit, got '$MODEL'" >&2
    exit 2
  fi
done

RUNS=()
for MODEL in "${MODELS[@]}"; do
  for G in "${GATES[@]}"; do
    for N in "${N_VALUES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        RUNS+=("$MODEL $G $N $SEED")
      done
    done
  done
done

FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "model gates N seed".
run_one() {
  set -uo pipefail
  local MODEL G N SEED FIXED RUN_DIR status
  read -r MODEL G N SEED <<< "$1"
  if [[ "$G" == "fixed" ]]; then FIXED=true; else FIXED=false; fi
  RUN_DIR="$OUT_DIR/${MODEL%_circuit}_${G}_N${N}_seed${SEED}"
  echo "--- model=$MODEL gates=$G N=$N seed=$SEED samples_per_p=$PER_P"
  uv run python experiments/haar_readouts/haar_readouts.py \
    "models=$MODEL" \
    "N=$N" \
    "seed=$SEED" \
    "$MODEL.fixed_instance=$FIXED" \
    "readouts_task.samples_per_p=$PER_P" \
    ${P_LIST:+"readouts_task.p_values=$P_LIST"} \
    ${R_LIST:+"readouts_task.readouts=$R_LIST"} \
    ${F_LIST:+"readouts_task.features=$F_LIST"} \
    "hydra.run.dir=$RUN_DIR"
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): model=$MODEL gates=$G N=$N seed=$SEED" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR PER_P P_LIST R_LIST F_LIST FAILED_LOG

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

echo "Done: ${#MODELS[@]} x ${#GATES[@]} x ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
