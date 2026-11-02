#!/usr/bin/env bash
# Purification decoding of a reference qubit by a ridge decoder of the record
# (experiments/QELM_R/exp_QELM_R.py, issue #247), swept over the dynamics
# (Haar, Floquet), N and the seed. Every run fits one decoder per measurement probability of
# purification_task.p_values.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_R/exp_QELM_R.sh
#
# Each run writes its arrays, metrics.json and Hydra config to
#     outputs/exp_QELM_R/<date_time>/<dynamics>_N<N>_seed<seed>/
# which git ignores; experiments/QELM_R/plot_QELM_R.py draws the figures from
# that folder. Environment variables change the defaults (lists are
# space-separated):
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=...                 write the sweep exactly there
#     DYNAMICS="haar floquet"     gates (purification_task.dynamics)
#     N_VALUES="6 8 10 12"        numbers of system qubits
#     SEEDS="42 43 44"            seeds (one draw of the Haar circuit and measurements)
#     PER_P=2000                  trajectories per p (purification_task.samples_per_p)
#     P_VALUES="0 0.1 0.2"        overrides purification_task.p_values
#     P_C_HAAR=0.17               p_c drawn on the figures for haar
#     P_C_FLOQUET=                p_c for floquet (empty: none drawn)
#     SCRAMBLE=0                  purification_task.scramble_rounds (default:
#                                 the config, 4 N); the run folders then end
#                                 in _scr<SCRAMBLE>, so that one sweep can
#                                 hold several values
#     PROTOCOL=paper              purification_task.protocol (default: the
#                                 config, steady); paper also closes the chain
#                                 (purification_task.periodic=true) and the
#                                 run folders end in _paper
#     ALPHA=1.0                   fixed ridge regularization (default: chosen
#                                 by cross-validation, training.alpha=null)
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
EXPERIMENT="exp_QELM_R"
read -r -a DYNAMICS <<< "${DYNAMICS:-haar floquet}"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10 12}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
PER_P="${PER_P:-2000}"
P_C_HAAR="${P_C_HAAR:-0.17}"
P_C_FLOQUET="${P_C_FLOQUET:-}"
MAX_JOBS="${MAX_JOBS:-1}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
# P_VALUES, when set, becomes the list purification_task.p_values=[...].
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
for DYN in "${DYNAMICS[@]}"; do
  for N in "${N_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUNS+=("$DYN $N $SEED")
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

# One run of the grid, called by xargs with its line "dynamics N seed".
run_one() {
  set -uo pipefail
  local DYN N SEED P_C RUN_DIR status
  read -r DYN N SEED <<< "$1"
  RUN_DIR="$OUT_DIR/${DYN}_N${N}_seed${SEED}${SCRAMBLE:+_scr$SCRAMBLE}${PROTOCOL:+_$PROTOCOL}"
  if [[ "$DYN" == "floquet" ]]; then P_C="$P_C_FLOQUET"; else P_C="$P_C_HAAR"; fi
  echo "--- dynamics=$DYN N=$N seed=$SEED samples_per_p=$PER_P p_c=${P_C:-none}"
  uv run python experiments/QELM_R/exp_QELM_R.py \
    "purification_task.dynamics=$DYN" \
    "N=$N" \
    "seed=$SEED" \
    "purification_task.samples_per_p=$PER_P" \
    ${P_LIST:+"purification_task.p_values=$P_LIST"} \
    ${P_C:+"purification_task.p_c=$P_C"} \
    ${ALPHA:+"training.alpha=$ALPHA"} \
    ${SCRAMBLE:+"purification_task.scramble_rounds=$SCRAMBLE"} \
    ${PROTOCOL:+"purification_task.protocol=$PROTOCOL"} \
    ${PROTOCOL:+"purification_task.periodic=$([[ $PROTOCOL == paper ]] && echo true || echo false)"} \
    "hydra.run.dir=$RUN_DIR"
  status=$?
  if (( status != 0 )); then
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): dynamics=$DYN N=$N seed=$SEED" >&2
    return 1
  fi
}
export -f run_one
export OUT_DIR PER_P P_LIST P_C_HAAR P_C_FLOQUET FAILED_LOG ALPHA SCRAMBLE PROTOCOL

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

echo "Done: ${#DYNAMICS[@]} x ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
