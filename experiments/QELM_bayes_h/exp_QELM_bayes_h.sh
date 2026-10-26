#!/usr/bin/env bash
# Learnability zone of the unitary QELM along h (issue #194): runs
# experiments/QELM_bayes_h/exp_QELM_bayes_h.py for every N, transverse field h
# and seed, then draws the accuracy and the mean entropy of the chance, Ridge
# and Bayes (QDA) classifiers against h with plot_bayes_h.py, one figure per N.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_bayes_h/exp_QELM_bayes_h.sh
#
# Each run writes its arrays, metrics.json, its log and its Hydra config to
#     outputs/QELM_bayes_h/<date_time>/N<N>_h<h>_seed<seed>/
# and the figures land in outputs/QELM_bayes_h/<date_time>/.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there
#     L=2000                      samples per run (samples.L)
#     N_VALUES="4 6 8"            total numbers of qubits, ancilla included
#     H_VALUES="0 1 10"           transverse fields TFIM.h, in units of TFIM.J = 1
#     T=10                        total evolution time evolution.t, in 1/J
#     SEEDS="0 1 2"               seeds (one disorder and label draw each)
#     INITIAL=uniform             sector states: uniform or product

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

# Each run is small: one thread is faster than several (issue #179).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="QELM_bayes_h"
read -r -a N_VALUES <<< "${N_VALUES:-4 6 8}"
read -r -a H_VALUES <<< "${H_VALUES:-0 0.1 0.2 0.3 0.5 0.75 1 1.5 2 3 4 5 6 8 10}"
T="${T:-10}"
L="${L:-2000}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
INITIAL="${INITIAL:-uniform}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebooks, issue #151) wins over the
# default folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# h and t are always passed, so that .hydra/config.yaml holds their numbers
# rather than the ${mul:...} and ${div:...} formulas.
for N in "${N_VALUES[@]}"; do
  for h in "${H_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_DIR="$OUT_DIR/N${N}_h${h}_seed${SEED}"
      echo "--- N=$N TFIM.h=$h t=$T seed=$SEED samples.L=$L initial=$INITIAL"
      uv run python experiments/QELM_bayes_h/exp_QELM_bayes_h.py \
        "N=$N" \
        "TFIM.h=$h" \
        "evolution.t=$T" \
        "seed=$SEED" \
        "samples.L=$L" \
        "ancilla_charge_sector.initial=$INITIAL" \
        "hydra.run.dir=$RUN_DIR"
    done
  done
done

uv run python experiments/QELM_bayes_h/plot_bayes_h.py "$OUT_DIR"

echo "Done: ${#N_VALUES[@]} x ${#H_VALUES[@]} x ${#SEEDS[@]} runs in $OUT_DIR"
