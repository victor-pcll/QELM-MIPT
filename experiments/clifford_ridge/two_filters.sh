#!/usr/bin/env bash
# Runs of issue #349 (two-filter decomposition). About 1 h 30 min on 3 cores.
set -euo pipefail
OUT=${OUT:-outputs/two_filters}
P="0.02 0.04 0.08 0.12 0.16 0.2 0.25 0.3 0.4 0.5"
PL="0.02 0.04 0.08 0.12 0.16 0.2 0.25 0.3 0.4"
run() { uv run python experiments/clifford_ridge/two_filters.py "$@" --out "$OUT"; }
{
  run exact --n 4 --samples 40 --k-max 4 --p $P
  run exact --n 6 --samples 16 --k-max 4 --p $P
  run shots --samples 20 --k-max 2 --shots 4 16 64 256 --p $P
} > "$OUT.exact.log" 2>&1 &
run locality --n 8 --samples 200 --k-max 4 --p $P > "$OUT.loc8.log" 2>&1 &
run locality --n 12 --samples 60 --k-max 3 --p $P > "$OUT.loc12.log" 2>&1 &
{
  run locality --n 16 --samples 100 --k-max 3 --p $P
  run locality --n 20 --samples 30 --k-max 3 --p $PL
} > "$OUT.loc1620.log" 2>&1 &
wait
