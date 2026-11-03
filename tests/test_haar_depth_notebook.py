"""notebooks/results/09_haar_depth_scaling.ipynb runs end to end (#268).

The code cells are executed in order, as Jupyter would, with the sweep shrunk
through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "09_haar_depth_scaling.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N_VALUES": "3 4",
        "DEPTHS": "3 N",
        "SEEDS": "1 2",
        "PER_P": "40",
        "P_VALUES": "0 0.3 0.6",
        "MAX_JOBS": "2",
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
    assert len(namespace["rows"]) == 8
    assert set(namespace["summary"]["rules"]) == {"3", "N"}
    # Profiles, crossings, cost, edges and conditioning.
    assert len(shows) == 5
    figure_path = namespace["figure_path"]
    assert figure_path.parent.parent == tmp_path / "haar_depth"  # #151
    for name in ("depth_profiles", "depth_crossings", "depth_cost", "depth_edges"):
        assert (figure_path.parent / f"{name}.png").is_file()
