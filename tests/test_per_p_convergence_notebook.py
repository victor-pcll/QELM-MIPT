"""notebooks/results/09_per_p_convergence.ipynb runs end to end (#267).

The code cells are executed in order, as Jupyter would, with the sweep shrunk
through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "09_per_p_convergence.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key, value in {
        "N_MAIN": "4",
        "N_VALUES": "3 4",
        "P_VALUES": "0.05 0.2 0.4",
        "REGIMES": "0.05 0.4",
        "PER_P_VALUES": "20 40 80",
        "N_CIRCUITS": "2",
        "N_JOBS": "1",
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOK.parent)  # where Jupyter starts

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)

    assert Path.cwd() == REPO_ROOT
    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "per_p_convergence"
    assert sorted(namespace["runs"]) == [3, 4]
    assert set(namespace["main"]["summary"]["by_p"]) == {"0.05", "0.2", "0.4"}
    # Profile, steps, heatmap, spectrum, scaling.
    assert len(shows) == 5
    for name in (
        "convergence_profile",
        "convergence_steps",
        "convergence_heatmap",
        "convergence_spectrum",
        "per_p_scaling",
    ):
        assert (sweep_dir / f"{name}.png").is_file()
        assert (sweep_dir / f"{name}.json").is_file()
    assert (sweep_dir / "decision.json").is_file()
