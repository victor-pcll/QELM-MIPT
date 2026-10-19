#!/usr/bin/env bash
# Projective-readout experiment (experiments/projective/projective_readout.py) on a small
# sweep over N and the seed.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/projective/projective_readout.sh
#
# Each run writes its arrays, metrics.json, log and Hydra config to
#     outputs/projective_readout/<date_time>/N<N>_seed<seed>/
# which git ignores. Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm   write under /tmp/qelm instead of outputs
#     L=1000               number of samples per run

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
EXPERIMENT="projective_readout"
MODEL="TFIM"
N_VALUES=(4 6)       # number of qubits, ancilla included
SEEDS=(0)            # one run per seed and per N
L="${L:-200}"        # samples per run (samples.L)
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="$OUT_ROOT/$EXPERIMENT/$STAMP"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$OUT_DIR/N${N}_seed${SEED}"
    echo "--- models=$MODEL $MODEL.N=$N seed=$SEED samples.L=$L"
    uv run python experiments/projective/projective_readout.py \
      "models=$MODEL" \
      "$MODEL.N=$N" \
      "seed=$SEED" \
      "samples.L=$L" \
      "hydra.run.dir=$RUN_DIR"
  done
done

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
