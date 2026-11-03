#!/usr/bin/env bash
# KL and Jensen-Shannon divergences between P(M | Q+) and P(M | Q-) (issue
# #270): runs experiments/kl_divergence/kl_divergence.py for every chain
# length, seed and measurement probability p, then draws the figures with
# plot_kl_divergence.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/kl_divergence/kl_divergence.sh
#
# Each run writes the log-likelihood ratios, metrics.json, its log and its
# Hydra config to
#     outputs/kl_divergence/<date_time>/N<N>_seed<seed>_p<p>/
# and the figures land in outputs/kl_divergence/<date_time>/.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there
#     L=8000                      records of the Haar-averaged observer
#     N_C=20                      fixed circuits (0 skips that observer)
#     PER_CIRCUIT=1000            records per fixed circuit
#     N_VALUES="6 8 10"           chain lengths
#     SEEDS="0"                   independent sets of records
#     P_VALUES="0.05 0.1"         measurement probabilities
#     JOBS=4                      runs in parallel (default 1)
#     P_C=0.175                   dashed line of the figures
#     ENCODING=local              charge (default) or local: the reference
#                                 qubit in a Bell pair with the middle site
#     GATES=u4                    u1 (default) or u4 (Haar U(4))

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# Find the QELM package even when Python skips the editable install's .pth
# file (macOS can mark it hidden, see issue #80).
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Use this project's .venv even when another virtualenv is active (#79).
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="kl_divergence"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10}"
L="${L:-8000}"
N_C="${N_C:-20}"
PER_CIRCUIT="${PER_CIRCUIT:-1000}"
read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a P_VALUES <<< "${P_VALUES:-0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4}"
JOBS="${JOBS:-1}"
P_C="${P_C:-0.175}"
ENCODING="${ENCODING:-charge}"
GATES="${GATES:-u1}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

run_one() {
  local N="$1" SEED="$2" P="$3"
  local RUN_DIR="$OUT_DIR/N${N}_seed${SEED}_p${P}"
  echo "--- N=$N seed=$SEED p=$P samples.L=$L n_circuits=$N_C"
  uv run python experiments/kl_divergence/kl_divergence.py \
    "N=$N" \
    "seed=$SEED" \
    "samples.L=$L" \
    "kl.p=$P" \
    "kl.n_circuits=$N_C" \
    "kl.records_per_circuit=$PER_CIRCUIT" \
    "kl.encoding=$ENCODING" \
    "kl.gates=$GATES" \
    "hydra.run.dir=$RUN_DIR" > "$OUT_DIR/N${N}_seed${SEED}_p${P}.log" 2>&1 \
    || { cat "$OUT_DIR/N${N}_seed${SEED}_p${P}.log"; return 1; }
  echo "done N=$N seed=$SEED p=$P"
}

pids=()
for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for P in "${P_VALUES[@]}"; do
      run_one "$N" "$SEED" "$P" &
      pids+=("$!")
      # Keep at most JOBS runs at once.
      while (( $(jobs -rp | wc -l) >= JOBS )); do
        wait -n
      done
    done
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status )); then
  echo "A run failed, see the logs in $OUT_DIR" >&2
  exit 1
fi

uv run python experiments/kl_divergence/plot_kl_divergence.py "$OUT_DIR" --p-c "$P_C"

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} x ${#P_VALUES[@]} runs in $OUT_DIR"
