"""notebooks/benchmarks/04_reference_qubit_purification.ipynb runs end to end (#185).

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
NOTEBOOK = (
    REPO_ROOT / "notebooks" / "benchmarks" / "04_reference_qubit_purification.ipynb"
)


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "L": "30",
        "N_VALUES": "4 8 16",
        "P_VALUES": "0.1 0.2 0.3",
        "SEEDS": "0",
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
    assert sorted(namespace["curves"]) == [4, 8, 16]
    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "purification_gullans_huse"
    assert namespace["figure_path"].is_file()
    assert namespace["figure_path"].with_suffix(".json").is_file()
    fss = json.loads((sweep_dir / "fss.json").read_text())
    assert "collapse_N>=16" not in fss  # one size >= 16: no fit
    assert fss["collapse_all"]["sizes"] == [4, 8, 16]
    assert len(namespace["fig"].axes) == 4
    assert len(shows) == 1
    os.chdir(REPO_ROOT)
