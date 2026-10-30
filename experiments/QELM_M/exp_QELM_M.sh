#!/usr/bin/env bash
# QELM experiment on the measurement record (experiments/QELM_M/exp_QELM_M.py),
# swept over N, the transverse field h, the seed and the measurement
# probability p (issues #94, #174).
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_M/exp_QELM_M.sh
#
# Each run writes its arrays, metrics.json, log and Hydra config to
#     outputs/exp_QELM_M/<date_time>/N<N>_seed<seed>_p<p>/
# or N<N>_h<h>_seed<seed>_p<p>/ when H_VALUES is set,
# which git ignores; notebooks/02_QELM_M_analysis.ipynb reads them back.
# Environment variables change the defaults (lists are space-separated):
#     OUT_ROOT=/tmp/qelm      write under /tmp/qelm instead of outputs
#     L=1000                  number of samples per run
#     ALPHA=1e-2              ridge regularization (training.alpha, default 1e-3)
#     N_VALUES="4 6"          numbers of system qubits, ancilla excluded
#     SEEDS="42 43 44"        seeds (one disorder and measurement draw each)
#     P_VALUES="0.1 0.5 1.0"  measurement probabilities
#     H_VALUES="1 10"         transverse fields TFIM.h, in units of TFIM.J = 1
#                             (issue #174); unset keeps the field of the
#                             config (10 J). TFIM only.
#     MODEL=haar_circuit      unitary dynamics: TFIM (default), haar_circuit
#                             or clifford_circuit (issue #171)

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
MODEL="${MODEL:-TFIM}"
read -r -a N_VALUES <<< "${N_VALUES:-4}"   # system qubits, ancilla excluded (issue #235)
L="${L:-12}"                                # samples per run (samples.L)
# ALPHA, when set, overrides training.alpha; otherwise the default of
# conf/training/regression/ridge.yaml (1e-3) applies (issue #92).
read -r -a SEEDS <<< "${SEEDS:-42}"
read -r -a P_VALUES <<< "${P_VALUES:-0.1 0.3 0.5 0.8 1.0}"
# TFIM.h, swept only when H_VALUES is set (issue #174). One empty entry
# otherwise: the loop runs once, without the override, and the run folders
# keep their old names. A circuit has no TFIM block, hence no field.
read -r -a H_VALUES <<< "${H_VALUES:-}"
if [[ ${#H_VALUES[@]} -eq 0 ]]; then
  H_VALUES=("")
elif [[ "$MODEL" != "TFIM" ]]; then
  echo "H_VALUES sets TFIM.h, it needs MODEL=TFIM, not $MODEL." >&2
  exit 1
fi
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebooks, issue #151) wins over the
# default folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# The measurement probability is projective_measurement.p, the only p the
# code reads (issue #90). An unmeasured qubit counts as 0 in the features
# (issue #91).

for N in "${N_VALUES[@]}"; do
  for h in "${H_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for p in "${P_VALUES[@]}"; do
        RUN_DIR="$OUT_DIR/N${N}${h:+_h$h}_seed${SEED}_p${p}"
        echo "--- models=$MODEL N=$N TFIM.h=${h:-config default} seed=$SEED samples.L=$L training.alpha=${ALPHA:-config default} p=$p"
        uv run python experiments/QELM_M/exp_QELM_M.py \
          "models=$MODEL" \
          "N=$N" \
          ${h:+"TFIM.h=$h"} \
          "seed=$SEED" \
          "samples.L=$L" \
          ${ALPHA:+"training.alpha=$ALPHA"} \
          "projective_measurement.p=$p" \
          "hydra.run.dir=$RUN_DIR"
      done
    done
  done
done

echo "Done: ${#N_VALUES[@]} x ${#H_VALUES[@]} x ${#SEEDS[@]} x ${#P_VALUES[@]} runs in $OUT_DIR"
