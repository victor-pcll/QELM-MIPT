"""notebooks/benchmarks/01_learnability_accuracy.ipynb runs end to end (#113).

The code cells are executed in order, as Jupyter would, with the sweep
shrunk through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "benchmarks" / "01_learnability_accuracy.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "L": "20",
        "N_VALUES": "4 6",
        "SEEDS": "0 1",
        "P_VALUES": "0.0 1.0",
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
    table = namespace["table"]
    assert [(r["N"], r["p"], r["n_runs"]) for r in table] == [
        (4, 0.0, 2),
        (4, 1.0, 2),
        (6, 0.0, 2),
        (6, 1.0, 2),
    ]
    # Exact limits: blind guess at p = 0, certainty at p = 1.
    for r in table:
        assert r["accuracy_mean"] == pytest.approx(0.5 if r["p"] == 0 else 1.0)
    assert namespace["figure_path"].is_file()
    assert namespace["figure_path"].with_suffix(".json").is_file()  # #147
    # Its own folder, holding its own figure only (#151).
    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "learnability_fig2b"
    assert not (sweep_dir / "learnability_fig2c.png").exists()
    assert "N4-N6" in namespace["comparison"]["binder_crossings"]["P_corr_binder"]
    assert len(shows) == 1
    os.chdir(REPO_ROOT)
