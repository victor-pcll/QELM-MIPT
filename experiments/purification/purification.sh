#!/usr/bin/env bash
# Reference-qubit purification benchmark against Gullans & Huse, PRL 125,
# 070606 (2020), Fig. 1 (issue #185): runs experiments/purification/purification.py
# for every N, p and seed, then draws the figure with plot_purification.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/purification/purification.sh
#
# Each run writes S_Q, metrics.json, its log and its Hydra config to
#     outputs/purification_gullans_huse/<date_time>/N<N>_p<p>_seed<seed>/
# and the figure, its data and fss.json land in the sweep folder.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there (issue #151)
#     L=2000                      trajectories (circuits) per run (samples.L)
#     N_VALUES="8 16 32 64"       system qubits, the paper's L, even
#     P_VALUES="0.10 ... 0.22"    measurement probabilities
#     SEEDS="0"                   several seeds split the trajectories of a point
#     MODES="reference"           probes: reference (purification.py, S_Q) and/or
#                                 steady (steady_state.py, S_half and I3, #257)
#     MAX_JOBS=1                  runs at once, one core each (JOBS, the
#                                 former name, works when MAX_JOBS is unset)

# Stop at the first failing command, unset variable or failing pipe.
set -euo pipefail

# Work from the repository root so that `uv run` and the paths below resolve,
# whatever directory the script is launched from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Find the QELM package in src/ even when Python skips the editable install's
# .pth file (macOS can mark it hidden, see issue #80).
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run with this project's .venv even when another project's virtualenv is
# active (issue #79).
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="purification_gullans_huse"
read -r -a N_VALUES <<< "${N_VALUES:-8 16 32 64}"
read -r -a P_VALUES <<< "${P_VALUES:-0.10 0.12 0.14 0.15 0.16 0.17 0.18 0.20 0.22}"
read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a MODES <<< "${MODES:-reference}"
L="${L:-2000}"
MAX_JOBS="${MAX_JOBS:-${JOBS:-1}}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got $MAX_JOBS." >&2
  exit 1
fi
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

run_one() {
  local N="$1" P="$2" SEED="$3" MODE="$4" SCRIPT RUN_NAME
  case "$MODE" in
    reference) SCRIPT=purification.py; RUN_NAME="N${N}_p${P}_seed${SEED}" ;;
    steady) SCRIPT=steady_state.py; RUN_NAME="N${N}_p${P}_steady_seed${SEED}" ;;
    *) echo "Unknown mode $MODE (reference or steady)." >&2; return 1 ;;
  esac
  uv run python "experiments/purification/$SCRIPT" \
    "N=$N" \
    "probe.p=$P" \
    "seed=$SEED" \
    "samples.L=$L" \
    "hydra.run.dir=$OUT_DIR/$RUN_NAME" > /dev/null 2>&1 \
    || { echo "FAILED: N=$N p=$P seed=$SEED mode=$MODE" >&2; return 1; }
  echo "--- done N=$N p=$P seed=$SEED mode=$MODE"
}

# Largest sizes first, so that the parallel runs end together.
SORTED_N=($(printf '%s\n' "${N_VALUES[@]}" | sort -rn))
PIDS=()
for N in "${SORTED_N[@]}"; do
  for P in "${P_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for MODE in "${MODES[@]}"; do
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
          sleep 0.2
        done
        run_one "$N" "$P" "$SEED" "$MODE" &
        PIDS+=("$!")
      done
    done
  done
done
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
if (( STATUS )); then
  echo "Some runs failed, see above." >&2
  exit 1
fi

if [[ " ${MODES[*]} " == *" reference "* ]]; then
  uv run python experiments/purification/plot_purification.py "$OUT_DIR"
fi
uv run python experiments/purification/plot_estimators.py "$OUT_DIR"

echo "Done: ${#N_VALUES[@]} x ${#P_VALUES[@]} x ${#SEEDS[@]} x ${#MODES[@]} runs in $OUT_DIR"
