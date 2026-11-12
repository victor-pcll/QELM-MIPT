#!/usr/bin/env bash
# Full sweep of the effect spectrum and ridge bias/variance (issue #334).
#
# Level I: N = 4 (20 seeds, 5000 records) and N = 6 (8 seeds, 6000 records).
# Level II: N = 6 (8 seeds), 8 (4 seeds) with 256 shots per input,
# N = 10 (4 seeds) with 64 shots, N = 12 (2 seeds) with 16 shots and fewer
# inputs. JOBS runs in parallel (default 4); OUT_ROOT sets the output folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/ridge_spectrum"
JOBS="${JOBS:-4}"
SCRIPT=experiments/ridge_spectrum/ridge_spectrum.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
{
  for s in $(seq 0 3); do echo "level2 --n 8 --seed $s --kmax 256"; done
  for s in $(seq 0 3); do echo "level2 --n 10 --seed $s --kmax 64"; done
  for s in 0 1; do echo "level2 --n 12 --seed $s --kmax 16 --sizes 500 250 250"; done
  for s in $(seq 0 7); do echo "level1 --n 6 --seed $s --records 6000"; done
  for s in $(seq 0 7); do echo "level2 --n 6 --seed $s --kmax 256"; done
  for s in $(seq 0 19); do echo "level1 --n 4 --seed $s --records 5000"; done
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python $SCRIPT {} --out $OUT >> $OUT/log.txt 2>&1"
echo "Done: outputs in $OUT"
