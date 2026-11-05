#!/usr/bin/env bash
# Convergence of the QELM ridge in PER_P (issue #267): one run of
# experiments/per_p_convergence/per_p_convergence.py per size N, with T = N
# rounds. The reference size N_MAIN runs every p of P_VALUES (the map
# accuracy(p, PER_P)); the other sizes run the three regimes only (the
# projection of PER_P* with N).
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/per_p_convergence/per_p_convergence.sh
#
# Each size writes per_p_convergence.npz, summary.json and its config to
#     outputs/per_p_convergence/<date_time>/N<N>/
# (or N<N>_c<ids>/ with CIRCUIT_IDS), then the figures are drawn in
# <date_time>/. Environment variables change the defaults (lists are
# space-separated):
#     OUT_ROOT=/scratch/qelm      write under /scratch/qelm instead of outputs
#     OUT_DIR=...                 write the sweep exactly there
#     N_MAIN=8                    reference size, every p of P_VALUES
#     N_VALUES="6 8 10"           sizes
#     P_VALUES="0.02 0.05 ..."    p of N_MAIN (default: conf/per_p_convergence.yaml)
#     REGIMES="0.05 0.18 0.4"     p of the other sizes
#     PER_P_VALUES="500 ... 32000"  overrides convergence.per_p_values
#     N_CIRCUITS=20               circuits of the ensemble
#     CIRCUIT_IDS="0 1"           only these circuits (a cluster job); merge
#                                 the jobs with merge_per_p_convergence.py
#     N_JOBS=0                    processes per run, one circuit each (0: all cores)
#     SEED=42                     seed of the circuits and the trajectories
#
# Cluster (SLURM), one job per circuit at N = 10, then the merge:
#
#     #SBATCH --array=0-19
#     #SBATCH --cpus-per-task=1
#     N_VALUES=10 N_MAIN=10 P_VALUES="0.05 0.18 0.4" CIRCUIT_IDS=$SLURM_ARRAY_TASK_ID \
#       N_JOBS=1 OUT_DIR=outputs/per_p_convergence/N10_cluster \
#       ./experiments/per_p_convergence/per_p_convergence.sh
#     uv run python experiments/per_p_convergence/merge_per_p_convergence.py \
#       outputs/per_p_convergence/N10_cluster/N10 outputs/per_p_convergence/N10_cluster/N10_c*

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
unset VIRTUAL_ENV
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

EXPERIMENT="per_p_convergence"
N_MAIN="${N_MAIN:-8}"
read -r -a N_VALUES <<< "${N_VALUES:-6 8 10}"
REGIMES="${REGIMES:-0.05 0.18 0.4}"
N_CIRCUITS="${N_CIRCUITS:-20}"
N_JOBS="${N_JOBS:-0}"
SEED="${SEED:-42}"

as_list() { echo "[$(echo "$1" | xargs | tr ' ' ',')]"; }

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

RUN_DIRS=()
for N in "${N_VALUES[@]}"; do
  ARGS=(
    "N=$N"
    "seed=$SEED"
    "convergence.n_rounds=$N"
    "convergence.n_circuits=$N_CIRCUITS"
    "convergence.n_jobs=$N_JOBS"
    "convergence.regimes=$(as_list "$REGIMES")"
  )
  if [[ "$N" == "$N_MAIN" ]]; then
    [[ -n "${P_VALUES:-}" ]] && ARGS+=("convergence.p_values=$(as_list "$P_VALUES")")
  else
    ARGS+=("convergence.p_values=$(as_list "$REGIMES")")
  fi
  [[ -n "${PER_P_VALUES:-}" ]] && ARGS+=("convergence.per_p_values=$(as_list "$PER_P_VALUES")")
  RUN_DIR="$OUT_DIR/N$N"
  if [[ -n "${CIRCUIT_IDS:-}" ]]; then
    ARGS+=("convergence.circuit_ids=$(as_list "$CIRCUIT_IDS")")
    RUN_DIR="$OUT_DIR/N${N}_c$(echo "$CIRCUIT_IDS" | xargs | tr ' ' '-')"
  fi
  echo "--- N=$N T=$N -> $RUN_DIR"
  uv run python experiments/per_p_convergence/per_p_convergence.py \
    "${ARGS[@]}" "hydra.run.dir=$RUN_DIR"
  RUN_DIRS+=("$RUN_DIR")
done

# A job over a subset of circuits leaves the figures to the merged run.
if [[ -z "${CIRCUIT_IDS:-}" ]]; then
  uv run python experiments/per_p_convergence/plot_per_p_convergence.py \
    "${RUN_DIRS[@]}" --out "$OUT_DIR"
fi
