"""run_live forwards a command's output while it runs (notebook tqdm bars)."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from QELM.utils import run_live


class Recorder(io.StringIO):
    """Keeps every write separately, to see that output arrives in pieces."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return super().write(text)


def test_returns_stdout_and_stderr_and_writes_them() -> None:
    out = Recorder()
    script = (
        "import sys; print('to stdout', flush=True); "
        "print('to stderr', file=sys.stderr)"
    )
    text = run_live([sys.executable, "-c", script], out=out)
    assert "to stdout\n" in text and "to stderr\n" in text
    assert out.getvalue() == text


def test_carriage_returns_reach_the_output_before_the_end() -> None:
    # A tqdm-like redraw: "\r" and no newline, then a pause before exiting.
    # The first redraw must be written before the command ends.
    script = (
        "import sys, time\n"
        "sys.stderr.write('\\r 1/2'); sys.stderr.flush(); time.sleep(0.5)\n"
        "sys.stderr.write('\\r 2/2\\n'); sys.stderr.flush()\n"
    )
    out = Recorder()
    text = run_live([sys.executable, "-c", script], out=out)
    assert text == "\r 1/2\r 2/2\n"
    assert out.writes[0] == "\r 1/2"


def test_tqdm_bar_is_forwarded() -> None:
    script = "from tqdm import tqdm\nfor _ in tqdm(range(3)): pass\n"
    text = run_live([sys.executable, "-c", script], out=io.StringIO())
    assert "3/3" in text


def test_env_and_cwd_are_passed(tmp_path) -> None:
    script = "import os; print(os.environ['QELM_LIVE'], os.getcwd())"
    text = run_live(
        [sys.executable, "-c", script],
        env={"QELM_LIVE": "yes", "PATH": ""},
        cwd=tmp_path,
        out=io.StringIO(),
    )
    assert text.split() == ["yes", str(tmp_path)]


def test_failure_raises_with_the_output() -> None:
    script = "import sys; print('boom', file=sys.stderr); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError) as info:
        run_live([sys.executable, "-c", script], out=io.StringIO())
    assert info.value.returncode == 3
    assert "boom" in info.value.output


def test_split_utf8_character_is_decoded() -> None:
    # "é" is two bytes; written one byte at a time it must still decode.
    script = (
        "import sys, time\n"
        "b = 'é\\n'.encode()\n"
        "for i in range(len(b)):\n"
        "    sys.stdout.buffer.write(b[i:i+1]); sys.stdout.flush(); time.sleep(0.05)\n"
    )
    assert run_live([sys.executable, "-c", script], out=io.StringIO()) == "é\n"
