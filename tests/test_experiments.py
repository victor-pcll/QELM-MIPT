"""The experiments/ entry point and its git-ignored scratch folder."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_experiment.py"
SCRATCH = REPO_ROOT / "experiments" / "scratch"


def test_layout_and_executable_bit() -> None:
    """Le script existe, est exécutable, et scratch/ est suivi via .gitkeep."""
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().startswith("#!/usr/bin/env python3\n")
    assert (SCRATCH / ".gitkeep").is_file()


def test_scratch_contents_are_ignored_but_gitkeep_is_not() -> None:
    """git ignore tout scratch/ sauf .gitkeep."""

    def ignored(path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False
        )
        return result.returncode == 0

    assert ignored("experiments/scratch/output.npy")
    assert ignored("experiments/scratch/run/log.txt")
    assert not ignored("experiments/scratch/.gitkeep")
    assert not ignored("experiments/run_experiment.py")


def test_script_runs_with_hydra_overrides(tmp_path: Path) -> None:
    """Le script compose conf/ avec les overrides, depuis n'importe quel dossier."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "models=kitaev", f"hydra.run.dir={tmp_path}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Running experiment with model: kitaev" in result.stdout
