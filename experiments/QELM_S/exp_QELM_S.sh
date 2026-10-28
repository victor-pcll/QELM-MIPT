#!/usr/bin/env bash
# Prediction of the half-chain entanglement entropy by the QELM
# (experiments/QELM_S/exp_QELM_S.py, issue #230), swept over the model, N and
# the seed. Every run pools all the measurement probabilities of
# entropy_task.p_values into one dataset.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_S/exp_QELM_S.sh
#
# Each run writes its arrays, metrics.json and Hydra config to
#     outputs/exp_QELM_S/<date_time>/<model>_N<N>_seed<seed>/
# which git ignores; experiments/QELM_S/plot_QELM_S.py draws the figures from
# that folder. Environment variables change the defaults (lists are
# space-separated):
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=...                 write the sweep exactly there
#     MODELS="haar_circuit TFIM"  dynamics (models=...)
#     N_VALUES="6 8 10"           numbers of qubits
#     SEEDS="42 43"               seeds (one draw of dynamics and measurements)
#     PER_P=200                   trajectories per p (entropy_task.samples_per_p)
#     P_VALUES="0 0.1 0.2"        overrides entropy_task.p_values
#     P_C_HAAR=0.17               p_c of the phase label for haar_circuit
#     P_C_TFIM=                   p_c for the TFIM (empty: no classification)
#     P_C_FLOQUET=0.21            p_c for floquet_circuit (S_1 of the Floquet
#                                 model, Table I of Skinner et al. 2019)
#     P_C_U1=                     p_c for U1_circuit (empty: no classification)
#     N_ROUNDS=                   entropy_task.n_rounds (empty: 2 N)
#     ALPHA=1e-2                  ridge regularization (training.alpha)
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
EXPERIMENT="exp_QELM_S"
read -r -a MODELS <<< "${MODELS:-haar_circuit TFIM}"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10}"
read -r -a SEEDS <<< "${SEEDS:-42}"
PER_P="${PER_P:-200}"
P_C_HAAR="${P_C_HAAR:-0.17}"
P_C_TFIM="${P_C_TFIM:-}"
P_C_FLOQUET="${P_C_FLOQUET:-0.21}"
P_C_U1="${P_C_U1:-}"
N_ROUNDS="${N_ROUNDS:-}"
MAX_JOBS="${MAX_JOBS:-1}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
# P_VALUES, when set, becomes the list entropy_task.p_values=[...].
P_LIST=""
if [[ -n "${P_VALUES:-}" ]]; then
  P_LIST="[$(echo "$P_VALUES" | xargs | tr ' ' ',')]"
fi
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

RUNS=()
for MODEL in "${MODELS[@]}"; do
  for N in "${N_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUNS+=("$MODEL $N $SEED")
    done
  done
done
if (( ${#RUNS[@]} == 0 )); then
  echo "The grid is empty: every list of values must hold at least one" >&2
  exit 2
fi
DUPLICATES="$(printf '%s\n' "${RUNS[@]}" | sort | uniq -d)"
if [[ -n "$DUPLICATES" ]]; then
  echo "The grid holds the same run twice, which would share a folder:" >&2
  echo "$DUPLICATES" >&2
  exit 2
fi

FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "model N seed".
run_one() {
  set -uo pipefail
  local MODEL N SEED P_C RUN_DIR status
  read -r MODEL N SEED <<< "$1"
  RUN_DIR="$OUT_DIR/${MODEL}_N${N}_seed${SEED}"
  case "$MODEL" in
    TFIM) P_C="$P_C_TFIM" ;;
    floquet_circuit) P_C="$P_C_FLOQUET" ;;
    U1_circuit) P_C="$P_C_U1" ;;
    *) P_C="$P_C_HAAR" ;;
  esac
  echo "--- models=$MODEL N=$N seed=$SEED samples_per_p=$PER_P p_c=${P_C:-none}"
  uv run python experiments/QELM_S/exp_QELM_S.py \
    "models=$MODEL" \
    "N=$N" \
    "seed=$SEED" \
    "entropy_task.samples_per_p=$PER_P" \
    ${P_LIST:+"entropy_task.p_values=$P_LIST"} \
    ${P_C:+"entropy_task.p_c=$P_C"} \
    ${N_ROUNDS:+"entropy_task.n_rounds=$N_ROUNDS"} \
    ${ALPHA:+"training.alpha=$ALPHA"} \
    "hydra.run.dir=$RUN_DIR"
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): models=$MODEL N=$N seed=$SEED" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR PER_P P_LIST P_C_HAAR P_C_TFIM P_C_FLOQUET P_C_U1 N_ROUNDS \
  FAILED_LOG ALPHA

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

echo "Done: ${#MODELS[@]} x ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
