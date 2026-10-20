"""Run a sweep script with one progress bar for the whole sweep.

The QELM_M launcher runs one Python process per (N, seed, p), each with its
own tqdm bar over the samples. From a notebook, ``subprocess.run(...,
capture_output=True)`` shows nothing until the end. :func:`run_with_progress`
shows a single bar that moves by one at each finished run, turns the bars of
the runs off (``TQDM_DISABLE=1``), and keeps their output for the error
message if a run fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence

from tqdm import tqdm  # type: ignore[import-untyped]


def run_with_progress(
    cmd: Sequence[str],
    total: int,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    marker: str = "--- ",
    desc: str = "sweep",
    show: Sequence[str] = ("N", "seed", "p"),
) -> str:
    """Run ``cmd`` and show one bar of ``total`` steps, one per run.

    A run starts when the command prints a line beginning with ``marker``
    (the launcher prints ``--- models=... p=...`` before each run). The bar
    moves when the next run starts and, for the last run, when the command
    exits successfully. The ``key=value`` pairs of the marker line whose key
    is in ``show`` are written after the bar, for the run in progress.

    Args:
        cmd: Command and arguments, as for :func:`subprocess.run`.
        total: Number of runs, the length of the bar.
        env: Environment of the child; the current one if ``None``. The
            child's own tqdm bars are turned off with ``TQDM_DISABLE=1``.
        cwd: Working directory of the child.
        marker: Prefix of the line that announces a run.
        desc: Label of the bar.
        show: Keys of the marker line shown after the bar; all of the line
            if none of them appears.

    Returns:
        Everything the command printed, stdout and stderr interleaved.

    Raises:
        subprocess.CalledProcessError: If the command exits with a nonzero
            code; its ``output`` holds what was printed, for the traceback.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["TQDM_DISABLE"] = "1"
    lines: list[str] = []
    started = 0
    with (
        tqdm(total=total, desc=desc, unit="run", file=sys.stdout) as bar,
        subprocess.Popen(
            list(cmd),
            env=child_env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as proc,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if line.startswith(marker):
                if started:
                    bar.update(1)  # the previous run has finished
                started += 1
                bar.set_postfix_str(_run_label(line[len(marker) :], show))
        proc.wait()
        if proc.returncode == 0 and started:
            bar.update(1)
            bar.set_postfix_str("done")
    output = "".join(lines)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd), output)
    return output


def _run_label(text: str, show: Sequence[str]) -> str:
    """``"models=TFIM N=4 seed=42 p=0.1"`` -> ``"N=4 seed=42 p=0.1"``."""
    kept = [w for w in text.split() if w.partition("=")[0] in show]
    return " ".join(kept) if kept else text.strip()
