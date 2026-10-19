#!/usr/bin/env bash
# Simple QELM experiment (experiments/QELM_M/exp_QELM_M.py) on a small sweep over N and
# the seed, without measurement during the evolution.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_M/exp_QELM_M.sh
#
# Each run writes its arrays, metrics.json, log and Hydra config to
#     outputs/exp_QELM_M/<date_time>/N<N>_seed<seed>/
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

# Run with this project's .venv even when another project's virtualenv is
# active: uv ignores a foreign VIRTUAL_ENV anyway, but warns on every call
# (issue #79).
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="exp_QELM_M"
MODEL="TFIM"
N_VALUES=(4)          # number of qubits, ancilla included
L="${L:-12}"         # samples per run (samples.L)
# The default alpha=1 also shrinks the bias weight and costs R2; 1e-3 does not.
ALPHA="${ALPHA:-1e-3}"
SEED="42"
p_VALUES=(1.0 0.5)
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="$OUT_ROOT/$EXPERIMENT/$STAMP"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# The measurement probability is projective_measurement.p, the only p the
# code reads (dynamic_readout.p had no effect, issue #90). For p < 1 the
# unmeasured qubits give NaN outcomes and NaN metrics (issue #91).

for N in "${N_VALUES[@]}"; do
  for p in "${p_VALUES[@]}"; do
    RUN_DIR="$OUT_DIR/N${N}_seed${SEED}_p${p}"
    echo "--- models=$MODEL $MODEL.N=$N seed=$SEED samples.L=$L training.alpha=$ALPHA p=$p"
    uv run python experiments/QELM_M/exp_QELM_M.py \
      "models=$MODEL" \
      "$MODEL.N=$N" \
      "seed=$SEED" \
      "samples.L=$L" \
      "training.alpha=$ALPHA" \
      "projective_measurement.p=$p" \
      "hydra.run.dir=$RUN_DIR"
  done
done

echo "Done: ${#N_VALUES[@]} x ${#p_VALUES[@]} runs in $OUT_DIR"
