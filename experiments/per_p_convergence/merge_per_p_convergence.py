#!/usr/bin/env python3
"""Merge runs of per_p_convergence.py over different circuits (issue #267).

On a cluster, each job runs a subset of the circuits
(``convergence.circuit_ids``); this writes the merged run, as if a single
job had run them all:

    uv run python experiments/per_p_convergence/merge_per_p_convergence.py \\
        outputs/per_p_convergence/N10 outputs/per_p_convergence/N10_c*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from per_p_convergence import merge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", help="folder of the merged run")
    parser.add_argument("folders", nargs="+", help="run folders to merge")
    args = parser.parse_args()
    merge(args.folders, args.out_dir)


if __name__ == "__main__":
    main()
