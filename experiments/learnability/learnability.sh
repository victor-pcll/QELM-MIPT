#!/usr/bin/env bash
# Charge learnability against Barratt et al., PRL 129, 200602 (2022), Fig. 2
# (issues #113 and #115): runs experiments/learnability/learnability.py for
# every chain length, seed and measurement probability p, then overlays the
# accuracy (Fig. 2b) on the digitized reference with plot_learnability.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/learnability/learnability.sh
#
# Each run writes P_corr, the records, metrics.json, its log and its Hydra
# config to
#     outputs/learnability/<date_time>/N<N>_seed<seed>_p<p>/
# and the figure and comparison.json land in outputs/learnability/<date_time>/.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there (issue #151)
#     L=40000                     records per run (samples.L)
#     N_VALUES="6 8 10"           chain lengths (the paper's L; no ancilla)
#     SEEDS="0 1 2"               independent sets of records
#     P_VALUES="0.0 0.1 0.2"      measurement probabilities
#     FIGURES="2b"                figures to draw: 2b, 2c or both (default)

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
EXPERIMENT="learnability"
read -r -a N_VALUES <<< "${N_VALUES:-6}"   # chain length, the paper's L
L="${L:-40000}"                            # records per run, as in the paper
read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a P_VALUES <<< "${P_VALUES:-0.0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6}"
read -r -a FIGURES <<< "${FIGURES:-2b 2c}"
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
    for P in "${P_VALUES[@]}"; do
      RUN_DIR="$OUT_DIR/N${N}_seed${SEED}_p${P}"
      echo "--- N=$N seed=$SEED samples.L=$L p=$P"
      uv run python experiments/learnability/learnability.py \
        "N=$N" \
        "seed=$SEED" \
        "samples.L=$L" \
        "learnability.p=$P" \
        "hydra.run.dir=$RUN_DIR"
    done
  done
done

uv run python experiments/learnability/plot_learnability.py "$OUT_DIR" \
  --figures "${FIGURES[@]}"

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} x ${#P_VALUES[@]} runs in $OUT_DIR"
