#!/usr/bin/env bash
# Simple QELM experiment (experiments/exp_QELM.py) on a small sweep over N and
# the seed, without measurement during the evolution.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/scripts/exp_QELM.sh
#
# Each run writes its arrays, metrics.json, log and Hydra config to
#     outputs/exp_QELM/<date_time>/N<N>_seed<seed>/
# which git ignores. Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm   write under /tmp/qelm instead of outputs
#     L=1000               number of samples per run
#     ALPHA=1e-3           ridge regularization (training.alpha)

# Stop at the first failing command, unset variable or failing pipe.
set -euo pipefail

# Work from the repository root so that `uv run` and the paths below resolve,
# whatever directory the script is launched from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Find the QELM package in src/ even when Python skips the editable install's
# .pth file (macOS can mark it hidden, see issue #80).
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="exp_QELM_M"
MODEL="TFIM"
N_VALUES=(4)          # number of qubits, ancilla included
SEEDS=(42)            # one run per seed and per N
L="${L:-200}"         # samples per run (samples.L)
# The default alpha=1 also shrinks the bias weight and costs R2; 1e-3 does not.
ALPHA="${ALPHA:-1e-3}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="$OUT_ROOT/$EXPERIMENT/$STAMP"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$OUT_DIR/N${N}_seed${SEED}"
    echo "--- models=$MODEL $MODEL.N=$N seed=$SEED samples.L=$L training.alpha=$ALPHA"
    uv run python experiments/QELM_M/exp_QELM_M.py \
      "models=$MODEL" \
      "$MODEL.N=$N" \
      "seed=$SEED" \
      "samples.L=$L" \
      "training.alpha=$ALPHA" \
      "hydra.run.dir=$RUN_DIR"\
      "dynamic_readout.p=0.5"\
      "projective_measurement.p=0.5" 
  done
done

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
