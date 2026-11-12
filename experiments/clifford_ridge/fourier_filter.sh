#!/usr/bin/env bash
# Issue #341: the ridge on record features as an exact Fourier filter.
# JOBS runs in parallel (default 4); OUT_ROOT sets the output folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/fourier_filter"
JOBS="${JOBS:-4}"
S=experiments/clifford_ridge/fourier_filter.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
A="0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16"
B="0.18 0.2 0.22 0.25 0.3 0.35 0.4 0.5"
{
  echo "transfer --n 128 --samples 300 --p $A --tag a"
  echo "transfer --n 128 --samples 300 --p $B --tag b"
  echo "transfer --n 32 --samples 300 --pairs"
  echo "transfer --n 64 --samples 500"
  echo "transfer --n 16 --samples 1000 --pairs"
  echo "validate --n 16 --circuits 6"
  echo "validate --n 32 --circuits 6"
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python $S {} --out $OUT >> $OUT/log.txt 2>&1"
echo "Done: outputs in $OUT"
