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
        "N_VALUES": "3 4",
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
    assert "16/16" in capsys.readouterr().out  # one bar for the 16 runs
    assert len(namespace["rows"]) == 16
    # N = 3: 11 weights, so the ratios 0.5 and 2.0 give 6 and 22 training
    # samples, i.e. 6/11 and 2 after rounding. N = 4: 16 weights, 8 and 32
    # samples, exactly 0.5 and 2: the grid of ratios depends on N.
    table = namespace["table"]
    expected = {3: 6 / 11, 4: 0.5}
    assert [(r["N"], r["p"], r["ratio"], r["n_runs"]) for r in table] == [
        (N, p, pytest.approx(ratio), 2)
        for N in (3, 4)
        for p in (0.5, 1.0)
        for ratio in (expected[N], 2.0)
    ]
    # Curves against the ratio: one row of panels per N, N = 3 on top.
    axes = namespace["axes"]
    assert axes.shape == (2, len(namespace["METRICS"])) == (2, 3)
    for row, N in zip(axes, (3, 4), strict=True):
        for ax in row:
            assert ax.get_xscale() == "log"
            assert ax.get_title().endswith(f"N = {N}")
            # The interpolation threshold at alpha_ratio = 1.
            assert any(
                line.get_linestyle() == "--" and set(line.get_xdata()) == {1.0}
                for line in ax.lines
            )
        assert row[0].get_yscale() == "log"  # NMSE
    assert len(shows) == 2  # the curves, then the heatmaps
    figure_path = namespace["figure_path"]
    assert figure_path.parent.parent == tmp_path / "QELM_M_ratio"  # #151
    data = json.loads(figure_path.with_suffix(".json").read_text())
    assert len(data["panels"]) == 6
    assert len(data["extra"]["table"]) == 8

    # Heatmaps over (alpha_ratio, p), issue #164: one per metric and N,
    # alpha_ratio on the x axis and p on the y axis.
    heatmaps = [h[0] for h in namespace["HEATMAPS"]]
    assert heatmaps == ["nmse", "r2", "accuracy", "balanced_accuracy"]
    for N in (3, 4):
        for key in heatmaps:
            grid = namespace["grids"][(N, key)]
            np.testing.assert_allclose(grid.x, [expected[N], 2.0])
            np.testing.assert_array_equal(grid.y, [0.5, 1.0])
            assert np.all(np.isfinite(grid.mean)) and np.all(grid.n_runs == 2)
    axes_maps = namespace["axes_maps"]
    assert axes_maps.shape == (2, 4)
    for row, N in zip(axes_maps, (3, 4), strict=True):
        for ax in row:
            assert ax.get_xscale() == "log" and ax.get_yscale() == "linear"
            # Plain labels at the powers of 2 within the swept ratios (6/11 is
            # above 0.5), instead of the overlapping 10^k labels.
            ticks = [t.get_text() for t in ax.get_xticklabels()]
            assert ticks == {3: ["1", "2"], 4: ["0.5", "1", "2"]}[N]
            assert ax.get_title().endswith(f"N = {N}")
            # The interpolation threshold at alpha_ratio = 1, a vertical line.
            assert any(
                line.get_linestyle() == "--" and set(line.get_xdata()) == {1.0}
                for line in ax.lines
            )
    heatmap_path = namespace["heatmap_path"]
    assert heatmap_path.parent == figure_path.parent
    maps = json.loads(heatmap_path.with_suffix(".json").read_text())
    drawn = [h for panel in maps["panels"] for h in panel["heatmaps"]]
    assert len(drawn) == 8  # one per metric and N, none from the colorbars
    np.testing.assert_allclose(
        np.array(drawn[0]["values"]), namespace["grids"][(3, "nmse")].mean
    )
    np.testing.assert_allclose(
        np.array(drawn[4]["values"]), namespace["grids"][(4, "nmse")].mean
    )
    os.chdir(REPO_ROOT)
