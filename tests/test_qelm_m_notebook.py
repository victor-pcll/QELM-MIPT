"""notebooks/02_QELM_M_analysis.ipynb runs end to end on a small sweep (#94).

The code cells are executed in order, as Jupyter would, with the sweep
shrunk through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "02_QELM_M_analysis.ipynb"


def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    for key, value in {
        "L": "10",
        "N_VALUES": "3",
        "SEEDS": "42 43",
        "P_VALUES": "0.2 1.0",
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(REPO_ROOT / "notebooks")  # where Jupyter starts

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)

    assert Path.cwd() == REPO_ROOT
    assert len(namespace["rows"]) == 4
    assert [(r["p"], r["n_runs"]) for r in namespace["table"]] == [(0.2, 2), (1.0, 2)]
    assert namespace["figure_path"].is_file()
    assert namespace["figure_path"].parent.parent == tmp_path / "exp_QELM_M"
    os.chdir(REPO_ROOT)
