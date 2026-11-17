#!/usr/bin/env bash
# All runs of issue #358 (about 4 h on 4 cores). Big sizes are split over
# seeds so that the jobs balance; analyse.py merges the seeds.
set -euo pipefail
OUT=${OUT:-outputs/bipartite_entropy}
JOBS=${JOBS:-4}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="uv run python experiments/bipartite_entropy/run.py"
{
  for s in 0 1 2 3; do echo "$R haar --n 20 --samples 40 --stride 5 --seed $s --out $OUT"; done
  for s in 0 1 2; do echo "$R clifford --n 256 --samples 50 --seed $s --out $OUT"; done
  for s in 0 1; do echo "$R haar --n 18 --samples 250 --stride 2 --seed $s --out $OUT"; done
  for s in 0 1; do echo "$R clifford --n 128 --samples 200 --seed $s --out $OUT"; done
  echo "$R haar --n 16 --samples 1000 --stride 2 --out $OUT"
  echo "$R clifford --n 64 --samples 600 --out $OUT"
  echo "$R clifford --n 32 --samples 1000 --out $OUT"
  echo "$R clifford --n 16 --samples 1000 --out $OUT"
  for n in 14 12 10 8; do echo "$R haar --n $n --samples 2000 --out $OUT"; done
} | xargs -P "$JOBS" -I{} bash -c "{} > /dev/null"
