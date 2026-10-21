#!/usr/bin/env bash
# Sample-to-parameter ratio sweep of the QELM on the measurement record
# (experiments/QELM_M/exp_QELM_M_ratio.py, issue #96): N fixed, so the
# number of weights n_params = nbr_time_evolution * Ns + 1 is fixed, and the
# number of training samples n_train = alpha_ratio * n_params is swept.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_M/exp_QELM_M_ratio.sh
#
# Each run writes its arrays, metrics.json, config_used.yaml and Hydra config to
#     outputs/exp_QELM_M_ratio/<date_time>/N<N>_seed<seed>_p<p>_a<alpha_ratio>/
# which git ignores; notebooks/results/03_QELM_M_alpha_ratio.ipynb reads them
# back. Environment variables change the defaults (lists are space-separated):
#     OUT_ROOT=/tmp/qelm          write under /tmp/qelm instead of outputs
#     OUT_DIR=/tmp/qelm/sweep     write the runs exactly there (the notebook)
#     N_TEST=200                  test samples per run (experiment.n_test)
#     ALPHA=1e-3                  ridge penalty (training.alpha), not the ratio
#     N_VALUES="4"                total numbers of qubits, ancilla included
#     SEEDS="42 43 44"            seeds (one disorder and measurement draw each)
#     P_VALUES="0.5 1.0"          measurement probabilities
#     ALPHA_RATIOS="0.5 1.0 2.0"  training samples per weight
#     MAX_ATTEMPTS=2              tries per run before it is given up
#
# A sweep survives a crashing run (issue #168): a run that exits with an error
# (a segmentation fault is exit code 139) is tried again, up to MAX_ATTEMPTS
# times, then listed in $OUT_DIR/failed_runs.txt, and the sweep goes on. A run
# whose folder already holds config_used.yaml, the last file a run writes, is
# skipped, so launching again with the same OUT_DIR completes an interrupted
# sweep without redoing the finished runs.

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
EXPERIMENT="exp_QELM_M_ratio"
MODEL="TFIM"
read -r -a N_VALUES <<< "${N_VALUES:-4}"   # number of qubits, ancilla included
N_TEST="${N_TEST:-200}"                     # test samples (experiment.n_test)
# Ridge penalty, independent of the ratio. It must stay > 0 for ratios < 1.
ALPHA="${ALPHA:-1e-3}"
read -r -a SEEDS <<< "${SEEDS:-42}"
read -r -a P_VALUES <<< "${P_VALUES:-1.0}"
read -r -a ALPHA_RATIOS <<< "${ALPHA_RATIOS:-0.5 1.0 1.5 2.0 5.0 10.0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebooks, issue #151) wins over the
# default folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

FAILED_LIST="$OUT_DIR/failed_runs.txt"
rm -f "$FAILED_LIST"  # rewritten below: only the runs still failing
n_failed=0
n_skipped=0

for N in "${N_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for p in "${P_VALUES[@]}"; do
      for RATIO in "${ALPHA_RATIOS[@]}"; do
        RUN_DIR="$OUT_DIR/N${N}_seed${SEED}_p${p}_a${RATIO}"
        # The "--- " line announces a run (one step of the notebook's bar),
        # also for a skipped one.
        echo "--- models=$MODEL N=$N seed=$SEED p=$p alpha_ratio=$RATIO n_test=$N_TEST training.alpha=$ALPHA"
        if [[ -f "$RUN_DIR/config_used.yaml" ]]; then
          echo "already done, skipped: $RUN_DIR"
          n_skipped=$((n_skipped + 1))
          continue
        fi
        for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
          status=0
          uv run python experiments/QELM_M/exp_QELM_M_ratio.py \
            "models=$MODEL" \
            "N=$N" \
            "seed=$SEED" \
            "projective_measurement.p=$p" \
            "experiment.alpha_ratio=$RATIO" \
            "experiment.n_test=$N_TEST" \
            "training.alpha=$ALPHA" \
            "hydra.run.dir=$RUN_DIR" || status=$?
          if [[ $status -eq 0 ]]; then
            break
          fi
          echo "run failed with exit code $status (attempt $attempt of $MAX_ATTEMPTS): $RUN_DIR" >&2
        done
        if [[ $status -ne 0 ]]; then
          echo "$RUN_DIR exit code $status" >> "$FAILED_LIST"
          n_failed=$((n_failed + 1))
        fi
      done
    done
  done
done

echo "Done: ${#N_VALUES[@]} x ${#SEEDS[@]} x ${#P_VALUES[@]} x ${#ALPHA_RATIOS[@]} runs in $OUT_DIR ($n_skipped already done, $n_failed failed)"
if [[ $n_failed -gt 0 ]]; then
  echo "WARNING: $n_failed runs failed after $MAX_ATTEMPTS attempts, listed in $FAILED_LIST; launch again with OUT_DIR=$OUT_DIR to retry them" >&2
fi
