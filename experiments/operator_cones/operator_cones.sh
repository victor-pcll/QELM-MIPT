#!/usr/bin/env bash
# Operator light cones of monitored Haar U(4) circuits, OTOC and Krylov
# (issue #315): the four parts of experiments/operator_cones/operator_cones.py
# with the settings of the README, then the figures.
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/operator_cones/operator_cones.sh
#
# Environment variables change the defaults:
#     OUT_DIR=...      write there (default outputs/operator_cones/<date_time>)
#     JOBS=4           processes for the trajectories (default 4)
#     TRAJ_N8=400      trajectories per p at N = 8 (default 400)
#     TRAJ_N10=80      trajectories per p at N = 10 (default 80)
#
# Cost on 4 cores, about: chain 10 s, parker 1 min, trajectories N = 8
# 8 min and N = 10 45 min, Krylov N = 8 6 min and N = 6 1 min, one operator
# N <= 7 about 20 min, one branch N = 10 7 min and N = 12 about 90 min.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# One BLAS thread per process: the trajectories run JOBS processes at once.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset VIRTUAL_ENV

OUT_DIR="${OUT_DIR:-outputs/operator_cones/$(date +%Y-%m-%d_%H-%M-%S)}"
JOBS="${JOBS:-4}"
SCRIPT=experiments/operator_cones/operator_cones.py
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

uv run python "$SCRIPT" chain --out "$OUT_DIR"
uv run python "$SCRIPT" parker --out "$OUT_DIR"
uv run python "$SCRIPT" traj --N 8 --traj "${TRAJ_N8:-400}" --chunk 25 \
  --jobs "$JOBS" --out "$OUT_DIR"
uv run python "$SCRIPT" traj --N 10 --traj "${TRAJ_N10:-80}" --chunk 5 \
  --jobs "$JOBS" --out "$OUT_DIR"
uv run python "$SCRIPT" krylov --N 8 --out "$OUT_DIR"
uv run python "$SCRIPT" krylov --N 6 --out "$OUT_DIR"
uv run python "$SCRIPT" single --N 5 --rank --out "$OUT_DIR"
uv run python "$SCRIPT" single --N 6 --rank --out "$OUT_DIR"
uv run python "$SCRIPT" single --N 7 --out "$OUT_DIR"
uv run python "$SCRIPT" branch --N 10 --out "$OUT_DIR"
uv run python "$SCRIPT" branch --N 12 --p 0,0.05,0.17,0.35,0.65 --out "$OUT_DIR"
uv run python experiments/operator_cones/plot_operator_cones.py --data "$OUT_DIR"
echo "Done: $OUT_DIR"
