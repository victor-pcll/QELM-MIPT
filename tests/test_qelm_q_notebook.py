"""notebooks/results/04_QELM_Q_measurements.ipynb runs on a small sweep (#163).

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
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "04_QELM_Q_measurements.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N_VALUES": "3",
        "H_VALUES": "10 1",
        "SEEDS": "42 43",
        "P_VALUES": "0.0 1.0",
        "FINAL_VALUES": "false true",
        "TIME_N": "3",
        "T_VALUES": "0.5 10",
        "L": "10",
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
    out = capsys.readouterr().out
    assert "p = 0 recovers the standard QELM" in out
    assert "16/16" in out  # one bar for the 16 runs
    assert len(namespace["rows"]) == 16
    table = namespace["table"]
    assert sorted((r["h"], r["final"], r["p"], r["n_runs"]) for r in table) == [
        (h, final, p, 2) for h in (1, 10) for final in (False, True) for p in (0.0, 1.0)
    ]
    axes = namespace["axes"]
    # One row per setting of the final round, one column per metric.
    assert axes.shape == (2, len(namespace["METRICS"])) == (2, 3)
    # One curve per field on every panel.
    for ax in axes.flat:
        assert len(ax.containers) == 2
    # The main figure, one figure per observable for each final setting, then
    # the evolution-time sweep.
    assert len(shows) == 4
    for setting, path in namespace["observable_figures"].items():
        assert path.name == f"accuracy_by_observable_{setting}.png"
        obs = json.loads(path.with_suffix(".json").read_text())
        assert len(obs["panels"]) == 6
    figure_path = namespace["figure_path"]
    assert figure_path.parent.parent == tmp_path / "QELM_Q"  # #151
    data = json.loads(figure_path.with_suffix(".json").read_text())
    assert len(data["panels"]) == 6
    assert len(data["extra"]["table"]) == 8
    # The evolution-time sweep: its own folder, N x t x seed x p runs.
    assert "8/8" in out
    assert len(namespace["time_rows"]) == 8
    time_figure = namespace["time_figure"]
    assert time_figure.parent.parent == tmp_path / "QELM_Q_time"
    assert sorted({r["t"] for r in namespace["time_table"]}) == [0.5, 10]
    assert namespace["axes_t"].shape == (1, 4)
    os.chdir(REPO_ROOT)
