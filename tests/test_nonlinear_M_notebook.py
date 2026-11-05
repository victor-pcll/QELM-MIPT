"""notebooks/results/10_QELM_M_nonlinear_features.ipynb runs end to end (#269).

The code cells are executed in order, as Jupyter would, on a small sweep set
through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "10_QELM_M_nonlinear_features.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N_VALUES": "4",
        "SEEDS": "1 2",
        "PER_P": "30",
        "P_VALUES": "0 0.5 1",
        "MODES": "raw abs xor_local",
        "MLP": "false",
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
    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "nonlinear_M"
    assert namespace["summary"]["N4"]["n_circuits"] == 2
    assert all(path.is_file() for path in namespace["figure_paths"])
    assert len(shows) == 1
    os.chdir(REPO_ROOT)
