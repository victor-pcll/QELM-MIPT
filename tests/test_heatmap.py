"""Color maps of a two-parameter sweep, QELM.utils.plots.heatmap (issue #164)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

from QELM.utils import summarize
from QELM.utils.plots import (
    cell_edges,
    draw_heatmap,
    figure_data,
    save_figure,
    subplots,
    sweep_grid,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _summary() -> list[dict]:
    """N = 4 on p in {0, 0.5, 1} x ratio in {0.5, 1, 4}, two seeds, one hole.

    The metric is ``p + ratio`` plus ``+-0.1`` from the seed, so its mean
    is exactly ``p + ratio``. The point ``(p, ratio) = (1, 4)`` has no run.
    N = 6 has a single point.
    """
    rows = []
    for seed, shift in ((0, -0.1), (1, 0.1)):
        for p in (1.0, 0.0, 0.5):
            for ratio in (4.0, 0.5, 1.0):
                if (p, ratio) != (1.0, 4.0):
                    rows.append(
                        {"N": 4, "p": p, "ratio": ratio, "m": p + ratio + shift}
                    )
    rows.append({"N": 6, "p": 0.5, "ratio": 1.0, "m": 7.0})
    return summarize(rows, by=("N", "p", "ratio"), metrics=["m"])


def test_sweep_grid_puts_the_mean_on_the_grid() -> None:
    grid = sweep_grid(_summary(), "p", "ratio", "m", size=4)
    np.testing.assert_array_equal(grid.x, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(grid.y, [0.5, 1.0, 4.0])
    assert grid.mean.shape == grid.n_runs.shape == (3, 3)
    expected = grid.x[None, :] + grid.y[:, None]
    expected[2, 2] = np.nan  # (p, ratio) = (1, 4) has no run
    np.testing.assert_allclose(grid.mean, expected)
    assert grid.n_runs[2, 2] == 0
    assert np.all(np.delete(grid.n_runs.ravel(), 8) == 2)


def test_sweep_grid_keeps_one_size() -> None:
    grid = sweep_grid(_summary(), "p", "ratio", "m", size=6)
    assert grid.mean.tolist() == [[7.0]]
    with pytest.raises(ValueError, match="No row"):
        sweep_grid(_summary(), "p", "ratio", "m", size=8)


def test_sweep_grid_rejects_two_rows_on_one_point() -> None:
    # Without size, N = 4 and N = 6 both have (p, ratio) = (0.5, 1).
    with pytest.raises(ValueError, match="Two rows"):
        sweep_grid(_summary(), "p", "ratio", "m")


def test_cell_edges_center_every_value() -> None:
    np.testing.assert_allclose(cell_edges([0.0, 0.5, 1.0]), [-0.25, 0.25, 0.75, 1.25])
    np.testing.assert_allclose(cell_edges([2.0]), [1.5, 2.5])


def test_cell_edges_are_geometric_on_a_log_scale() -> None:
    edges = cell_edges([0.25, 1.0, 4.0], log=True)
    np.testing.assert_allclose(edges, [0.125, 0.5, 2.0, 8.0])
    # Equal width on screen: a constant ratio between neighbouring edges.
    np.testing.assert_allclose(edges[1:] / edges[:-1], 4.0)
    # Every value sits at the geometric middle of its cell.
    np.testing.assert_allclose(np.sqrt(edges[1:] * edges[:-1]), [0.25, 1.0, 4.0])


@pytest.mark.parametrize(
    ("centers", "log", "match"),
    [
        ([], False, "non-empty"),
        ([1.0, 1.0], False, "increasing"),
        ([0.0, 1.0], True, "> 0"),
    ],
)
def test_cell_edges_reject_bad_centers(
    centers: list[float], log: bool, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        cell_edges(centers, log=log)


def test_draw_heatmap_draws_the_grid_with_a_colorbar() -> None:
    grid = sweep_grid(_summary(), "p", "ratio", "m", size=4)
    fig, ax = subplots()
    mesh = draw_heatmap(ax, grid, label="m", log_y=True)
    assert ax.get_yscale() == "log" and ax.get_xscale() == "linear"
    values = mesh.get_array()
    assert values is not None
    assert np.ma.getmaskarray(values).reshape(3, 3)[2, 2]  # the hole is blank
    np.testing.assert_allclose(mesh.norm.vmin, 0.5)
    np.testing.assert_allclose(mesh.norm.vmax, 4.5)
    assert len(fig.axes) == 2  # the panel and its colorbar
    assert fig.axes[1].get_ylabel() == "m"


def test_draw_heatmap_log_color_masks_non_positive_values() -> None:
    grid = sweep_grid(_summary(), "p", "ratio", "m", size=4)
    fig, ax = subplots()
    mesh = draw_heatmap(ax, grid, log_color=True, vmin=0.1, vmax=10.0)
    assert isinstance(mesh.norm, LogNorm)
    assert (mesh.norm.vmin, mesh.norm.vmax) == (0.1, 10.0)


def test_save_figure_writes_the_heatmap_data(tmp_path: Path) -> None:
    grid = sweep_grid(_summary(), "p", "ratio", "m", size=4)
    fig, ax = subplots()
    draw_heatmap(ax, grid, label="m", log_y=True)
    assert len(figure_data(fig)["panels"][1]["heatmaps"]) == 0  # the colorbar
    save_figure(fig, tmp_path, "heat")
    data = json.loads((tmp_path / "heat.json").read_text())
    (heat,) = data["panels"][0]["heatmaps"]
    assert heat["label"] == "m"
    np.testing.assert_allclose(heat["x_edges"], cell_edges(grid.x))
    np.testing.assert_allclose(heat["y_edges"], cell_edges(grid.y, log=True))
    np.testing.assert_allclose(np.array(heat["values"], dtype=float), grid.mean)
