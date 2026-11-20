#!/usr/bin/env bash
# Minimal parity weights of the Clifford effects, N = 6 to 12 (about 1 h on 4 cores).
set -euo pipefail
OUT=${OUT:-outputs/readout_locality}
for spec in "6 400" "8 400" "10 300" "12 100"; do
    set -- $spec
    uv run python experiments/readout_locality/run.py --n "$1" --circuits "$2" --out "$OUT"
done
# Depth control: T = N, 4N, 8N at N = 8 and T = N, 4N at N = 10 (about 15 min).
for spec in "8 8" "8 32" "8 64" "10 10" "10 40"; do
    set -- $spec
    uv run python experiments/readout_locality/run.py --n "$1" --depth "$2" --circuits 200 --out "$OUT"
done
