"""Run a command and show its output while it runs, also in Jupyter.

``subprocess.run(..., capture_output=True)`` only returns the output at the
end, and without capture the child writes to the terminal that started
Jupyter, not to the cell. :func:`run_live` reads the child's stdout and
stderr as they come and writes them to ``sys.stdout``, so progress bars
(tqdm redraws its line with ``\\r``) update under the cell.
"""

from __future__ import annotations

import codecs
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO


def run_live(
    cmd: Sequence[str],
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    out: TextIO | None = None,
) -> str:
    """Run ``cmd``, forward its stdout and stderr live, return them as text.

    Args:
        cmd: Command and arguments, as for :func:`subprocess.run`.
        env: Environment of the child; the current one if ``None``.
        cwd: Working directory of the child.
        out: Where the output is written; ``sys.stdout`` (the cell, in
            Jupyter) if ``None``.

    Returns:
        Everything the command printed, stdout and stderr interleaved.

    Raises:
        subprocess.CalledProcessError: If the command exits with a nonzero
            code; its ``output`` holds what was printed.
    """
    stream = sys.stdout if out is None else out
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks: list[str] = []
    with subprocess.Popen(
        list(cmd),
        env=None if env is None else dict(env),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        # os.read returns as soon as bytes arrive, so a tqdm redraw (no
        # newline) shows at once instead of waiting for the end of a line.
        while data := os.read(fd, 4096):
            text = decoder.decode(data)
            chunks.append(text)
            stream.write(text)
            stream.flush()
    tail = decoder.decode(b"", final=True)
    if tail:
        chunks.append(tail)
        stream.write(tail)
    output = "".join(chunks)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd), output)
    return output
