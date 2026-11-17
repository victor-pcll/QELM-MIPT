#!/usr/bin/env bash
# Runs of issue #360. About 1 h on 4 cores.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_ROOT:-outputs}/r2_budget"
JOBS="${JOBS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
D=experiments/r2_budget
{
  for prior in iid flat; do
    echo "$D/run.py dense --n 4 --circuits 300 --targets 40 --prior $prior"
    echo "$D/run.py clifford --n 8 --k 1 2 3 --circuits 200 --targets 40 --prior $prior"
    echo "$D/run.py clifford --n 12 --k 1 2 3 --circuits 100 --targets 40 --prior $prior"
  done
  echo "$D/run.py dense --n 4 --depth 8 --no-final --ensemble sphere --method recursive --prior flat --circuits 300 --targets 40"
} | xargs -P "$JOBS" -I{} sh -c "uv run --locked python {} --out $OUT >> $OUT/log.txt 2>&1"
echo "Done: outputs in $OUT"
