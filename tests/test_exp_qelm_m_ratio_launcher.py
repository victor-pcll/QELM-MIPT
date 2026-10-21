"""experiments/QELM_M/exp_QELM_M_ratio.sh survives a failing run (issue #168).

A run that keeps failing is tried MAX_ATTEMPTS times, listed in
failed_runs.txt, and the sweep goes on; launching again into the same
OUT_DIR skips the finished runs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M_ratio.sh"


def _launch(out_dir: Path, ratios: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "N_VALUES": "3",
        "SEEDS": "42",
        "P_VALUES": "1.0",
        "ALPHA_RATIOS": ratios,
        "N_TEST": "10",
        "MAX_ATTEMPTS": "2",
        "OUT_DIR": str(out_dir),
        "TQDM_DISABLE": "1",
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, check=False
    )


def test_failed_run_is_retried_listed_and_skipped_on_resume(tmp_path: Path) -> None:
    # alpha_ratio = -1 is rejected by the experiment: a run that always fails.
    first = _launch(tmp_path, "0.5 -1 2.0")
    assert first.returncode == 0, first.stderr
    for ratio in ("0.5", "2.0"):
        assert (tmp_path / f"N3_seed42_p1.0_a{ratio}" / "config_used.yaml").is_file()
    assert "attempt 1 of 2" in first.stderr and "attempt 2 of 2" in first.stderr
    failed = (tmp_path / "failed_runs.txt").read_text().splitlines()
    assert len(failed) == 1 and "N3_seed42_p1.0_a-1 " in failed[0]
    assert "(0 already done, 1 failed)" in first.stdout
    assert first.stdout.count("--- ") == 3  # one step of the bar per run

    # Same OUT_DIR: the finished runs are skipped, the failed list rewritten.
    stamp = (tmp_path / "N3_seed42_p1.0_a0.5" / "metrics.json").stat().st_mtime_ns
    second = _launch(tmp_path, "0.5 2.0")
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("already done, skipped") == 2
    assert "(2 already done, 0 failed)" in second.stdout
    assert second.stdout.count("--- ") == 2
    assert not (tmp_path / "failed_runs.txt").exists()
    after = (tmp_path / "N3_seed42_p1.0_a0.5" / "metrics.json").stat().st_mtime_ns
    assert after == stamp  # not run again
