#!/usr/bin/env bash
# Test experiment: run the QELM pipeline on a small sweep over N and the seed.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/example/test_experiment.sh
#
# Each run writes its Hydra log and checkpoint to
#     outputs/test_experiment/<date_time>/N<N>_seed<seed>/
# (no "=" in folder names: Hydra rejects it inside an override value)
# which git ignores. Set OUT_ROOT to write somewhere else:
#     OUT_ROOT=/tmp/qelm ./experiments/example/test_experiment.sh
#
# To build your own experiment, copy this file, rename it, and edit the
# "Parameters" block and the overrides passed to `uv run` below. See
# experiments/scripts/README.md.

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
# active: uv ignores a foreign VIRTUAL_ENV anyway, but warns on every call
# (issue #79).
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="test_experiment"
MODEL="TFIM"
N_VALUES=(4 6)   # number of qubits, ancilla included
SEEDS=(0 1)      # one run per seed and per N
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="$OUT_ROOT/$EXPERIMENT/$STAMP"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$OUT_DIR/N${N}_seed${SEED}"
    echo "--- models=$MODEL N=$N seed=$SEED"
    # Any key of conf/ can be overridden here with key=value, exactly as on
    # the command line. hydra.run.dir sets where the log and checkpoint go.
    uv run python main.py \
      "models=$MODEL" \
      "N=$N" \
      "seed=$SEED" \
      "hydra.run.dir=$RUN_DIR"
  done
done

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
