"""run_with_progress shows one bar for a whole sweep (notebook, issue #117)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from QELM.utils import run_with_progress

# Three "runs", each announced by a "--- " line and running its own tqdm bar.
SWEEP = (
    "from tqdm import tqdm\n"
    "for p in (0.1, 0.5, 1.0):\n"
    "    print(f'--- models=TFIM N=4 samples.L=7 p={p}', flush=True)\n"
    "    for _ in tqdm(range(7)): pass\n"
    "print('Done: 1 x 1 x 3 runs')\n"
)


def test_one_bar_with_one_step_per_run(capsys: pytest.CaptureFixture[str]) -> None:
    output = run_with_progress([sys.executable, "-c", SWEEP], total=3)
    shown = capsys.readouterr().out
    assert "3/3" in shown
    assert "N=4 p=1.0" in shown  # the run in progress, after the bar
    assert "models=TFIM" not in shown
    assert "7/7" not in shown and "7/7" not in output  # runs' bars are off
    assert output.count("--- ") == 3 and "Done: 1 x 1 x 3 runs" in output


def test_bar_does_not_count_a_run_before_it_ends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The second run fails: only the first one is counted as finished.
    script = (
        "import sys\n"
        "print('--- run 1', flush=True)\n"
        "print('--- run 2', flush=True)\n"
        "print('Traceback: boom', file=sys.stderr); sys.exit(2)\n"
    )
    with pytest.raises(subprocess.CalledProcessError) as info:
        run_with_progress([sys.executable, "-c", script], total=2)
    assert info.value.returncode == 2
    assert "Traceback: boom" in info.value.output
    shown = capsys.readouterr().out
    assert "1/2" in shown and "2/2" not in shown


def test_env_and_cwd_are_passed(tmp_path) -> None:
    script = "import os; print(os.environ['QELM_LIVE'], os.getcwd())"
    output = run_with_progress(
        [sys.executable, "-c", script],
        total=1,
        env={"QELM_LIVE": "yes", "PATH": ""},
        cwd=tmp_path,
    )
    assert output.split() == ["yes", str(tmp_path)]


def test_child_tqdm_is_disabled_even_with_a_given_env() -> None:
    script = "import os; print(os.environ['TQDM_DISABLE'])"
    output = run_with_progress(
        [sys.executable, "-c", script], total=1, env={"TQDM_DISABLE": "0"}
    )
    assert output.strip() == "1"
