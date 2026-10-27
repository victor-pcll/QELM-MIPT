#!/usr/bin/env bash
# Double descent of the QELM on MNIST digits, 3 vs 5 by default (issue
# #181): runs
# experiments/double_descent/double_descent.py once per seed, then draws the
# test and train errors against alpha = n_train / n_features with
# plot_double_descent.py.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/double_descent/double_descent.sh
#
# The first run downloads MNIST to data/mnist/ (about 11 MB, gitignored).
# Each run writes its sweep over alpha, metrics.json, its log and its Hydra
# config to
#     outputs/double_descent/<date_time>/N<N>_c<classes>_seed<seed>/
# (c35 for the digits 3 and 5), and the figures (double_descent.png,
# confusion_matrices.png) and summary.json land in
# outputs/double_descent/<date_time>/.
# Environment variables change the defaults:
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=path/to/sweep       write the sweep exactly there (issue #151)
#     N=8                         qubits, one per principal component
#     SEEDS="0 1 2"               reservoirs (TFIM draws), pools and ELM layers
#     CLASSES="3 5"               digits to tell apart, two or more ("3 5 8")
#     RIDGE_ALPHA=0.0             ridge penalty training.alpha (0: no penalty)
#     DATASET=mnist               mnist, or digits (8x8, bundled with scikit-learn)
#     N_POOL=2000                 training pool
#     N_TEST=null                 test images (null: the whole test split)
#     N_REPEATS=20                training sets per ratio
#     ENCODING=rotation_encoding  state encoding of the images (issue #196):
#                                 rotation_encoding, angle_encoding,
#                                 dense_angle_encoding, uniform_bloch_encoding
#                                 or general_encoding
#     EXTRA_OVERRIDES="..."       Hydra overrides added to every run

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
EXPERIMENT="double_descent"
N="${N:-8}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a CLASSES <<< "${CLASSES:-3 5}"
RIDGE_ALPHA="${RIDGE_ALPHA:-0.0}"
DATASET="${DATASET:-mnist}"
N_POOL="${N_POOL:-2000}"
N_TEST="${N_TEST:-null}"
N_REPEATS="${N_REPEATS:-20}"
ENCODING="${ENCODING:-rotation_encoding}"
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebook, issue #151) wins over the default
# folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# Hydra list of the digits, [3,5,8], and their tag in the run folder, 358.
CLASS_LIST="[$(IFS=,; echo "${CLASSES[*]}")]"
CLASS_TAG="$(IFS=; echo "${CLASSES[*]}")"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="$OUT_DIR/N${N}_c${CLASS_TAG}_seed${SEED}"
  echo "--- N=$N seed=$SEED classes=$CLASS_TAG dataset=$DATASET encoding=$ENCODING"
  uv run python experiments/double_descent/double_descent.py \
    "N=$N" \
    "seed=$SEED" \
    "double_descent.dataset=$DATASET" \
    "double_descent.classes=$CLASS_LIST" \
    "training.alpha=$RIDGE_ALPHA" \
    "double_descent.n_pool=$N_POOL" \
    "double_descent.n_test=$N_TEST" \
    "double_descent.n_repeats=$N_REPEATS" \
    "double_descent.encoding=$ENCODING" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "hydra.run.dir=$RUN_DIR"
done

uv run python experiments/double_descent/plot_double_descent.py "$OUT_DIR"

echo "Done: ${#SEEDS[@]} runs in $OUT_DIR"
