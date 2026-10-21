#!/usr/bin/env bash
# QELM on the charge Q under intermediate Z measurements
# (experiments/QELM_Q/exp_QELM_Q.py, issue #163), swept over N, the
# transverse field h, the seed, the measurement probability p and whether a
# last round follows the last step (projective_measurement.final_measurement).
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_Q/exp_QELM_Q.sh
#
# Each run writes its arrays, metrics.json, log and Hydra config to
#     outputs/exp_QELM_Q/<date_time>/N<N>_h<h>_final<final>_seed<seed>_p<p>/
# which git ignores; notebooks/results/04_QELM_Q_measurements.ipynb reads
# them back. Environment variables change the defaults (lists are
# space-separated):
#     OUT_ROOT=/tmp/qelm      write under /tmp/qelm instead of outputs
#     OUT_DIR=...             write the sweep exactly there (the notebook)
#     L=1000                  number of samples per run
#     ALPHA=1e-2              ridge regularization (training.alpha, default 1e-3)
#     N_VALUES="4 6"          total numbers of qubits, ancilla included
#     H_VALUES="10 1"         transverse fields TFIM.h, in units of TFIM.J = 1
#     SEEDS="42 43 44"        seeds (one disorder and measurement draw each)
#     P_VALUES="0 0.5 1.0"    measurement probabilities
#     FINAL_VALUES="false true"  without / with a round after the last step

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
EXPERIMENT="exp_QELM_Q"
MODEL="TFIM"
read -r -a N_VALUES <<< "${N_VALUES:-4}"   # number of qubits, ancilla included
read -r -a H_VALUES <<< "${H_VALUES:-10}"  # TFIM.h, 10 J is the config default
L="${L:-200}"                               # samples per run (samples.L)
# ALPHA, when set, overrides training.alpha; otherwise the default of
# conf/training/regression/ridge.yaml (1e-3) applies (issue #92).
read -r -a SEEDS <<< "${SEEDS:-42}"
read -r -a P_VALUES <<< "${P_VALUES:-0.0 0.2 0.5 0.8 1.0}"
read -r -a FINAL_VALUES <<< "${FINAL_VALUES:-false}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebooks, issue #151) wins over the
# default folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# h is always passed, so that .hydra/config.yaml holds its number rather than
# the ${mul:...} formula and the notebook can read it back.
for N in "${N_VALUES[@]}"; do
  for h in "${H_VALUES[@]}"; do
    for FINAL in "${FINAL_VALUES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        for p in "${P_VALUES[@]}"; do
          RUN_DIR="$OUT_DIR/N${N}_h${h}_final${FINAL}_seed${SEED}_p${p}"
          echo "--- models=$MODEL N=$N TFIM.h=$h final=$FINAL seed=$SEED samples.L=$L training.alpha=${ALPHA:-config default} p=$p"
          uv run python experiments/QELM_Q/exp_QELM_Q.py \
            "models=$MODEL" \
            "N=$N" \
            "TFIM.h=$h" \
            "seed=$SEED" \
            "samples.L=$L" \
            ${ALPHA:+"training.alpha=$ALPHA"} \
            "projective_measurement.p=$p" \
            "projective_measurement.final_measurement=$FINAL" \
            "hydra.run.dir=$RUN_DIR"
        done
      done
    done
  done
done

echo "Done: ${#N_VALUES[@]} x ${#H_VALUES[@]} x ${#FINAL_VALUES[@]} x ${#SEEDS[@]} x ${#P_VALUES[@]} runs in $OUT_DIR"
