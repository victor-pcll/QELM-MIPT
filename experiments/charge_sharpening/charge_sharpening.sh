#!/usr/bin/env bash
# Charge-sharpening benchmark against Agrawal et al., PRX 14, 041012 (2024),
# Fig. 7 (issue #137): runs experiments/charge_sharpening/charge_sharpening.py
# for every N, seed and measurement strength gamma, then overlays the sweep on
# the digitized reference curves with plot_charge_sharpening.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/charge_sharpening/charge_sharpening.sh
#
# Each run writes S_R, the measurement record, metrics.json, its log and its
# Hydra config to
#     outputs/charge_sharpening/<date_time>/N<N>_seed<seed>_g<gamma>/
# and the figure and comparison.json land in outputs/charge_sharpening/<date_time>/.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there (issue #151)
#     L=10000                     trajectories per run (samples.L)
#     N_VALUES="7"                total qubits, ancilla included (L_paper = N - 1)
#     SEEDS="0 1 2 3 4"           one circuit instance per seed
#     GAMMA_VALUES="0.0 0.5 1.0"  measurement strengths
#     FIXED_INSTANCE=false        new gates for every trajectory

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
EXPERIMENT="charge_sharpening"
read -r -a N_VALUES <<< "${N_VALUES:-7}"   # paper's L = 6 system qubits + R
L="${L:-10000}"                            # trajectories per run, as in the paper
read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a GAMMA_VALUES <<< "${GAMMA_VALUES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
FIXED_INSTANCE="${FIXED_INSTANCE:-true}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebooks, issue #151) wins over the
# default folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for GAMMA in "${GAMMA_VALUES[@]}"; do
      RUN_DIR="$OUT_DIR/N${N}_seed${SEED}_g${GAMMA}"
      echo "--- N=$N seed=$SEED samples.L=$L gamma=$GAMMA fixed_instance=$FIXED_INSTANCE"
      uv run python experiments/charge_sharpening/charge_sharpening.py \
        "N=$N" \
        "seed=$SEED" \
        "samples.L=$L" \
        "weak_measurement.gamma=$GAMMA" \
        "U1_circuit.fixed_instance=$FIXED_INSTANCE" \
        "hydra.run.dir=$RUN_DIR"
    done
  done
done

uv run python experiments/charge_sharpening/plot_charge_sharpening.py "$OUT_DIR"

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} x ${#GAMMA_VALUES[@]} runs in $OUT_DIR"
