#!/usr/bin/env bash
# Minimal parity weights of the Clifford effects, N = 6 to 12 (about 1 h on 4 cores).
set -euo pipefail
OUT=${OUT:-outputs/readout_locality}
for spec in "6 400" "8 400" "10 300" "12 100"; do
    set -- $spec
    uv run python experiments/readout_locality/run.py --n "$1" --circuits "$2" --out "$OUT"
done
