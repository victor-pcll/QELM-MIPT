#!/usr/bin/env bash
# Rank of the effect space against p, fixed mask, one panel per circuit
# class (issue #313): experiments/effect_rank/effect_rank.py for each size N
# and seed, then the figures.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/effect_rank/effect_rank.sh
#
# Environment variables change the defaults (lists are space-separated):
#     OUT_DIR=outputs/effect_rank   where the runs and figures go
#     SIZES="3 4 5 6 7"             sizes N, all four classes (T = N)
#     SEEDS="$(seq 0 9)"            seeds, one circuit per class and one mask
#     LARGE_SIZES="8"               sizes run only for LARGE_KINDS
#     LARGE_KINDS="u1 clifford"     classes that fit in memory at LARGE_SIZES
#
# Cost on 4 cores, per seed and all 21 values of p: under 10 s up to N = 6,
# about 6 min at N = 7 (Haar dominates). At N = 8 Haar and Z2 reach ranks of
# order 10^4 on vectors of length 4^8: beyond 16 GB of memory.
# Runs that already have their JSON are skipped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
OUT_DIR="${OUT_DIR:-outputs/effect_rank}"
read -r -a SIZES <<< "${SIZES:-3 4 5 6 7}"
read -r -a SEEDS <<< "${SEEDS:-$(seq -s ' ' 0 9)}"
read -r -a LARGE_SIZES <<< "${LARGE_SIZES:-8}"
read -r -a LARGE_KINDS <<< "${LARGE_KINDS:-u1 clifford}"
SCRIPT=experiments/effect_rank/effect_rank.py

for n in "${SIZES[@]}"; do
  for s in "${SEEDS[@]}"; do
    [[ -f "$OUT_DIR/N${n}_seed${s}.json" ]] && continue
    uv run python "$SCRIPT" --n "$n" --seed "$s" --out "$OUT_DIR"
  done
done
tag="$(IFS=-; echo "${LARGE_KINDS[*]}")"
for n in "${LARGE_SIZES[@]}"; do
  for s in "${SEEDS[@]}"; do
    [[ -f "$OUT_DIR/N${n}_seed${s}_${tag}.json" ]] && continue
    uv run python "$SCRIPT" --n "$n" --seed "$s" --kinds "${LARGE_KINDS[@]}" \
      --out "$OUT_DIR"
  done
done
uv run python experiments/effect_rank/plot_effect_rank.py "$OUT_DIR" \
  --out "$OUT_DIR/figures"
