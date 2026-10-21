"""notebooks/results/03_QELM_M_alpha_ratio.ipynb runs end to end on a small sweep.

Issue #96 for the curves against the ratio, #164 for the heatmaps over
``(p, alpha_ratio)``.

The code cells are executed in order, as Jupyter would, with the sweep
shrunk through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "03_QELM_M_alpha_ratio.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N_VALUES": "3",
        "SEEDS": "42 43",
        "P_VALUES": "0.5 1.0",
        "ALPHA_RATIOS": "0.5 2.0",
        "N_TEST": "10",
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
    assert "8/8" in capsys.readouterr().out  # one bar for the 8 runs
    assert len(namespace["rows"]) == 8
    # N = 3: 11 weights, so the ratios 0.5 and 2.0 give 6 and 22 training
    # samples, i.e. 6/11 and 2 after rounding.
    table = namespace["table"]
    assert [(r["p"], r["ratio"], r["n_runs"]) for r in table] == [
        (0.5, pytest.approx(6 / 11), 2),
        (0.5, 2.0, 2),
        (1.0, pytest.approx(6 / 11), 2),
        (1.0, 2.0, 2),
    ]
    axes = namespace["axes"]
    assert len(axes) == len(namespace["METRICS"]) == 3
    for ax in axes:
        assert ax.get_xscale() == "log"
        # The interpolation threshold at alpha_ratio = 1.
        assert any(
            line.get_linestyle() == "--" and set(line.get_xdata()) == {1.0}
            for line in ax.lines
        )
    assert axes[0].get_yscale() == "log"  # NMSE
    assert len(shows) == 2  # the curves, then the heatmaps
    figure_path = namespace["figure_path"]
    assert figure_path.parent.parent == tmp_path / "QELM_M_ratio"  # #151
    data = json.loads(figure_path.with_suffix(".json").read_text())
    assert len(data["panels"]) == 3
    assert len(data["extra"]["table"]) == 4

    # Heatmaps over (p, ratio), issue #164: one per metric.
    heatmaps = [h[0] for h in namespace["HEATMAPS"]]
    assert heatmaps == ["nmse", "r2", "accuracy", "balanced_accuracy"]
    for key in heatmaps:
        grid = namespace["grids"][(3, key)]
        np.testing.assert_array_equal(grid.x, [0.5, 1.0])
        np.testing.assert_allclose(grid.y, [6 / 11, 2.0])
        assert np.all(np.isfinite(grid.mean)) and np.all(grid.n_runs == 2)
    axes_maps = namespace["axes_maps"]
    assert axes_maps.shape == (1, 4)
    for ax in axes_maps.ravel():
        assert ax.get_xscale() == "linear" and ax.get_yscale() == "log"
        # The interpolation threshold at alpha_ratio = 1.
        assert any(
            line.get_linestyle() == "--" and set(line.get_ydata()) == {1.0}
            for line in ax.lines
        )
    heatmap_path = namespace["heatmap_path"]
    assert heatmap_path.parent == figure_path.parent
    maps = json.loads(heatmap_path.with_suffix(".json").read_text())
    drawn = [h for panel in maps["panels"] for h in panel["heatmaps"]]
    assert len(drawn) == 4  # one per metric, none from the colorbars
    np.testing.assert_allclose(
        np.array(drawn[0]["values"]), namespace["grids"][(3, "nmse")].mean
    )
    os.chdir(REPO_ROOT)
