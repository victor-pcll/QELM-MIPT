#!/usr/bin/env bash
# Full sweep of the effect spectrum and ridge bias/variance (issue #334).
#
# Level I: N = 4 (20 seeds, 5000 records) and N = 6 (8 seeds, 6000 records).
# Level II: N = 6 (8 seeds), 8 (4 seeds) with 256 shots per input, and a
# reduced N = 10 run (4 seeds, 3 values of p, 32 shots, 600 / 200 / 200
# inputs): with 64 shots and 1000 / 500 / 500 inputs one point of N = 10
# takes about 20 minutes on one core of four busy ones.
# JOBS runs in parallel (default 4); OUT_ROOT sets the output folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/ridge_spectrum"
JOBS="${JOBS:-4}"
SCRIPT=experiments/ridge_spectrum/ridge_spectrum.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
{
  for s in $(seq 0 3); do echo "level2 --n 8 --seed $s --kmax 256"; done
  for s in $(seq 0 3); do
    echo "level2 --n 10 --seed $s --kmax 32 --sizes 600 200 200 --p 0.1 0.25 0.4"
  done
  for s in $(seq 0 7); do echo "level1 --n 6 --seed $s --records 6000"; done
  for s in $(seq 0 7); do echo "level2 --n 6 --seed $s --kmax 256"; done
  for s in $(seq 0 19); do echo "level1 --n 4 --seed $s --records 5000"; done
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python $SCRIPT {} --out $OUT >> $OUT/log.txt 2>&1"
echo "Done: outputs in $OUT"
