#!/usr/bin/env bash
# Issue #352: entanglement of the backward trajectories, Haar (N = 8 to 16)
# and Clifford (N = 8, 12, 16) gates, plus N = 4 to 7 (both classes, the sizes
# of the exact ridge span of #339), in the geometry of #334/#339.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/haar_order"
JOBS="${JOBS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
D=experiments/clifford_ridge/haar_order.py
P7="0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.8 1"
{
  for n in 4 5 6 7; do
    for g in haar clifford; do
      echo "$D --n $n --samples 1000 --gates $g --p $P7 --out $OUT"
    done
  done
  echo "$D --n 16 --samples 300 --out $OUT"
  echo "$D --n 16 --samples 300 --gates clifford --out $OUT"
  echo "$D --n 14 --samples 500 --out $OUT"
  echo "$D --n 12 --samples 1000 --out $OUT"
  echo "$D --n 12 --samples 500 --gates clifford --out $OUT"
  echo "$D --n 10 --samples 1000 --out $OUT"
  echo "$D --n 8 --samples 1000 --out $OUT"
  echo "$D --n 8 --samples 500 --gates clifford --out $OUT"
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python {} >> $OUT/log.txt 2>&1"
echo "Done"
