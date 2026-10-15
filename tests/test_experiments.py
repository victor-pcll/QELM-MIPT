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


BASH_SCRIPT = REPO_ROOT / "experiments" / "scripts" / "test_experiment.sh"


def test_bash_script_is_executable_and_parses() -> None:
    """Le script bash de test est exécutable et sa syntaxe est valide."""
    assert BASH_SCRIPT.is_file()
    assert os.access(BASH_SCRIPT, os.X_OK)
    assert BASH_SCRIPT.read_text().startswith("#!/usr/bin/env bash\n")
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)


def test_bash_script_writes_one_checkpoint_per_run(tmp_path: Path) -> None:
    """Le script lance N in (4, 6) x seed in (0, 1) et écrit 4 checkpoints."""
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={**os.environ, "OUT_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 2 x 2 runs" in result.stdout
    runs = sorted(tmp_path.glob("test_experiment/*/*"))
    assert [run.name for run in runs] == [
        "N4_seed0",
        "N4_seed1",
        "N6_seed0",
        "N6_seed1",
    ]
    for run in runs:
        assert (run / "checkpoint.npz").is_file()
        assert (run / ".hydra" / "config.yaml").is_file()
