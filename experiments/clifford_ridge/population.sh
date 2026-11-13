#!/usr/bin/env bash
# Ridge at K = infinity on the Clifford QELM (issue #337): the group of the
# effect for a fresh circuit per sample, T = 2N, N = 16 to 128.
# The p grid is split in three chunks per N (one job each, tags c0..c2).
# JOBS runs in parallel (default 4); OUT_ROOT sets the output folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/clifford_ridge"
JOBS="${JOBS:-4}"
SCRIPT=experiments/clifford_ridge/clifford_ridge.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
C0="0.02 0.04 0.06 0.08 0.09 0.1 0.11 0.12 0.13 0.14"
C1="0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24"
C2="0.25 0.26 0.27 0.28 0.29 0.3 0.35 0.4 0.5"
{
  for spec in "128 200" "64 500" "32 1000" "16 2000"; do
    set -- $spec
    echo "population --n $1 --samples $2 --p $C0 --tag c0"
    echo "population --n $1 --samples $2 --p $C1 --tag c1"
    echo "population --n $1 --samples $2 --p $C2 --tag c2"
  done
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python $SCRIPT {} --out $OUT >> $OUT/log_population.txt 2>&1"
echo "Done: outputs in $OUT"
