#!/usr/bin/env bash
# QELM on the charge Q under intermediate Z measurements
# (experiments/QELM_Q/exp_QELM_Q.py, issue #163), swept over N, the
# transverse field h, the seed, the measurement probability p and whether a
# last round follows the last step (projective_measurement.final_measurement).
#
# Run from anywhere, after `uv sync`:
#
#     ./experiments/QELM_Q/exp_QELM_Q.sh
#
# Each run writes its arrays, metrics.json, log and Hydra config to
#     outputs/exp_QELM_Q/<date_time>/N<N>_h<h>_t<t>_final<final>_seed<seed>_p<p>/
# which git ignores; notebooks/results/04_QELM_Q_measurements.ipynb reads
# them back. Environment variables change the defaults (lists are
# space-separated):
#     OUT_ROOT=/tmp/qelm      write under /tmp/qelm instead of outputs
#     OUT_DIR=...             write the sweep exactly there (the notebook)
#     L=1000                  number of samples per run
#     DELTA=200               samples per system qubit: each run gets
#                             L = int(DELTA * N), and L is ignored (issue #201)
#     ALPHA=1e-2              ridge regularization (training.alpha, default 1e-3)
#     N_VALUES="4 6"          numbers of system qubits, ancilla excluded
#     H_VALUES="10 1"         transverse fields TFIM.h, in units of TFIM.J = 1
#     T_VALUES="0.5 2 10"     total evolution times evolution.t, in units of 1/J
#     SEEDS="42 43 44"        seeds (one disorder and measurement draw each)
#     P_VALUES="0 0.5 1.0"    measurement probabilities
#     FINAL_VALUES="false true"  without / with a round after the last step
#     BALANCED=true           exactly L / 2 samples per charge sector
#                             (samples.balanced, issue #233)
#     STATE=vector            simulate the state vector (state_representation,
#                             issue #183), much faster from N = 7 on
#     MAX_JOBS=4              runs at once (default 1), one core each; the
#                             runs wait in a queue (xargs -P, issue #199).
#                             At N = 11 (12 qubits with the ancilla), about
#                             1-1.5 GB per run with the density matrix: 6 to
#                             8 on 16 GB of RAM.
#                             JOBS is the former name and still works.
#
# A failed run (a Python error, or a run killed out of memory) does not stop
# the others: the sweep goes to the end, lists the failed runs in
# $OUT_DIR/failed_runs.txt (or FAILED_LOG) and exits with 1.

# Stop at the first failing command, unset variable or failing pipe.
set -euo pipefail

# Work from the repository root so that `uv run` and the paths below resolve,
# whatever directory the script is launched from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Find the QELM package in src/ even when Python skips the editable install's
# .pth file (macOS can mark it hidden, see issue #80).
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run with this project's .venv even when another project's virtualenv is
# active (issue #79).
unset VIRTUAL_ENV

# ---- Parameters -------------------------------------------------------------
EXPERIMENT="exp_QELM_Q"
MODEL="TFIM"
read -r -a N_VALUES <<< "${N_VALUES:-4}"   # system qubits, ancilla excluded (issue #235)
read -r -a H_VALUES <<< "${H_VALUES:-10}"  # TFIM.h, 10 J is the config default
read -r -a T_VALUES <<< "${T_VALUES:-10}"  # evolution.t, 10 / J is the config default
L="${L:-200}"                               # samples per run (samples.L)
# ALPHA, when set, overrides training.alpha; otherwise the default of
# conf/training/regression/ridge.yaml (1e-3) applies (issue #92).
read -r -a SEEDS <<< "${SEEDS:-42}"
read -r -a P_VALUES <<< "${P_VALUES:-0.0 0.2 0.5 0.8 1.0}"
read -r -a FINAL_VALUES <<< "${FINAL_VALUES:-false}"
# BALANCED, when set, overrides samples.balanced (false by default, #233).
# STATE, when set, overrides state_representation (density by default,
# issue #183).
# MAX_JOBS: runs at once (default 1). JOBS, its former name (#183), still
# works when MAX_JOBS is not set.
MAX_JOBS="${MAX_JOBS:-${JOBS:-1}}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got '$MAX_JOBS'" >&2
  exit 2
fi
# -----------------------------------------------------------------------------

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-outputs}"
# OUT_DIR set by the caller (the notebooks, issue #151) wins over the
# default folder, so that the caller knows where the sweep lands.
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$EXPERIMENT/$STAMP}"
mkdir -p "$OUT_DIR"
echo "Writing outputs to $OUT_DIR"

# The grid, one line "N h t final seed p" per run, in the order of the loops.
RUNS=()
for N in "${N_VALUES[@]}"; do
  for h in "${H_VALUES[@]}"; do
    for T in "${T_VALUES[@]}"; do
      for FINAL in "${FINAL_VALUES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
          for p in "${P_VALUES[@]}"; do
            RUNS+=("$N $h $T $FINAL $SEED $p")
          done
        done
      done
    done
  done
done

# An empty list of values gives no run; "${RUNS[@]}" would then be
# "unbound" for set -u in bash 3.2.
if (( ${#RUNS[@]} == 0 )); then
  echo "The grid is empty: every list of values must hold at least one" >&2
  exit 2
fi

# Each run writes only into its own folder (hydra.run.dir), so parallel runs
# never touch the same file, as long as two runs never get the same folder:
# that happens only if a list repeats a value, which is refused here.
DUPLICATES="$(printf '%s\n' "${RUNS[@]}" | sort | uniq -d)"
if [[ -n "$DUPLICATES" ]]; then
  echo "The grid holds the same run twice, which would share a folder:" >&2
  echo "$DUPLICATES" >&2
  exit 2
fi

# The failed runs, one line each. Workers append to it at the same time: an
# append of one short line (>>, below PIPE_BUF) is atomic, so lines never mix.
# FAILED_LOG, when set, puts it elsewhere: one per Slurm array task, since
# the tasks share OUT_DIR (experiments/README.md).
FAILED_LOG="${FAILED_LOG:-$OUT_DIR/failed_runs.txt}"
mkdir -p "$(dirname "$FAILED_LOG")"
: > "$FAILED_LOG"

# One run of the grid, called by xargs with its line "N h t final seed p".
# It runs in its own bash process, so it only sees exported scalars.
run_one() {
  set -uo pipefail
  local N h T FINAL SEED p RUN_DIR L_RUN status
  read -r N h T FINAL SEED p <<< "$1"
  RUN_DIR="$OUT_DIR/N${N}_h${h}_t${T}_final${FINAL}_seed${SEED}_p${p}"
  # With DELTA, L = int(DELTA * N), as exp_QELM_Q computes it: both samples.L
  # and samples.delta land in the run's config (#201).
  if [[ -n "${DELTA:-}" ]]; then
    L_RUN="$(awk -v d="$DELTA" -v n="$N" 'BEGIN { printf "%d", d * n }')"
  else
    L_RUN="$L"
  fi
  echo "--- models=$MODEL N=$N TFIM.h=$h t=$T final=$FINAL seed=$SEED samples.L=$L_RUN delta=${DELTA:-none} training.alpha=${ALPHA:-config default} p=$p"
  # h and t are always passed, so that .hydra/config.yaml holds their numbers
  # rather than the ${mul:...} and ${div:...} formulas and the notebook can
  # read them back.
  uv run python experiments/QELM_Q/exp_QELM_Q.py \
    "models=$MODEL" \
    "N=$N" \
    "TFIM.h=$h" \
    "evolution.t=$T" \
    "seed=$SEED" \
    "samples.L=$L_RUN" \
    ${DELTA:+"samples.delta=$DELTA"} \
    ${ALPHA:+"training.alpha=$ALPHA"} \
    "projective_measurement.p=$p" \
    "projective_measurement.final_measurement=$FINAL" \
    ${STATE:+"state_representation=$STATE"} \
    ${BALANCED:+"samples.balanced=$BALANCED"} \
    "hydra.run.dir=$RUN_DIR"
  status=$?
  if (( status != 0 )); then
    # 137 = killed by SIGKILL, what the kernel's OOM killer sends.
    echo "exit $status: $1" >> "$FAILED_LOG"
    echo "!!! run failed (exit $status): N=$N TFIM.h=$h t=$T final=$FINAL seed=$SEED p=$p" >&2
    # Exit 1 whatever the code: xargs keeps going after 1-125, and stops the
    # whole queue on 255 or if the worker itself dies from a signal.
    return 1
  fi
}
export -f run_one
export OUT_DIR MODEL L FAILED_LOG DELTA ALPHA STATE BALANCED

# A bounded queue: xargs starts the runs in order and keeps at most MAX_JOBS
# of them alive (issue #199). A failed run does not stop the others; xargs
# exits with 123 at the end if any failed. On a cluster the same grid maps to
# a Slurm job array, see experiments/README.md.
QUEUE_STATUS=0
printf '%s\0' "${RUNS[@]}" \
  | xargs -0 -P "$MAX_JOBS" -I {} bash -c 'run_one "$1"' _ {} \
  || QUEUE_STATUS=$?

if [[ -s "$FAILED_LOG" ]]; then
  echo "$(wc -l < "$FAILED_LOG" | tr -d ' ') of ${#RUNS[@]} runs failed, listed in $FAILED_LOG:" >&2
  cat "$FAILED_LOG" >&2
  exit 1
fi
rm -f "$FAILED_LOG"
if (( QUEUE_STATUS != 0 )); then
  echo "xargs stopped with exit $QUEUE_STATUS" >&2
  exit "$QUEUE_STATUS"
fi

echo "Done: ${#N_VALUES[@]} x ${#H_VALUES[@]} x ${#T_VALUES[@]} x ${#FINAL_VALUES[@]} x ${#SEEDS[@]} x ${#P_VALUES[@]} runs in $OUT_DIR"
