#!/usr/bin/env bash
# Runs of issue #356. About 1 h 30 min on 4 cores.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/dictionary_spectrum"
JOBS="${JOBS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
D=experiments/dictionary_spectrum
{
  echo "$D/clifford_weights.py --n 12 --samples 1000"
  echo "$D/clifford_weights.py --n 16 --samples 1000"
  echo "$D/clifford_weights.py --n 20 --samples 400"
  echo "$D/clifford_weights.py --n 24 --samples 60"
  for s in $(seq 0 7); do echo "$D/haar_spectrum.py --n 6 --seed $s --records 6000"; done
  for s in $(seq 0 19); do echo "$D/haar_spectrum.py --n 4 --seed $s"; done
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python {} --out $OUT >> $OUT/log.txt 2>&1"
echo "Done: outputs in $OUT"
