#!/usr/bin/env bash
# Issue #339: Haar against Clifford at K = infinity (N = 4 to 7, geometry of
# #334) and the entropy profile S(l) of the Clifford effect group
# (N = 16 to 256, T = 2N). JOBS runs in parallel (default 4).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}"
JOBS="${JOBS:-4}"
D=experiments/clifford_ridge
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/haar_span" "$OUT/clifford_ridge"
C0="0.02 0.04 0.06 0.08 0.09 0.1 0.11 0.12 0.13 0.14"
C1="0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24"
C2="0.25 0.26 0.27 0.28 0.29 0.3 0.35 0.4 0.5"
P7="0 0.05 0.1 0.15 0.2 0.3 0.4 0.6 0.8 1"
{
  for s in 0 1 2; do echo "$D/haar_span.py --n 7 --seed $s --p $P7 --out $OUT/haar_span"; done
  for c in "$C0 --tag c0" "$C1 --tag c1" "$C2 --tag c2"; do
    echo "$D/clifford_ridge.py locality --n 256 --samples 150 --p $c --out $OUT/clifford_ridge"
  done
  for n in 128 64 32 16; do
    echo "$D/clifford_ridge.py locality --n $n --samples 400 --out $OUT/clifford_ridge"
  done
  for s in $(seq 0 9); do echo "$D/haar_span.py --n 6 --seed $s --out $OUT/haar_span"; done
  for n in 4 5; do
    for s in $(seq 0 19); do echo "$D/haar_span.py --n $n --seed $s --out $OUT/haar_span"; done
  done
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python {} >> $OUT/log_issue339.txt 2>&1"
echo "Done"
