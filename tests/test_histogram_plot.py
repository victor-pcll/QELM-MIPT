"""Superimposed histograms of QELM.utils.plots.histogram (issue #191)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import BarContainer

from QELM.utils.plots import (
    color_map,
    common_edges,
    draw_histograms,
    save_figure,
    subplots,
)
from QELM.utils.plots.histogram import POINT_HALF_WIDTH

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_common_edges_span_every_sample() -> None:
    edges = common_edges([np.array([0.0, 1.0]), np.array([-2.0, 0.5])], bins=6)
    np.testing.assert_allclose(edges, np.linspace(-2, 1, 7))


def test_common_edges_of_a_constant_feature() -> None:
    edges = common_edges([np.zeros(5), np.zeros(3)], bins=2)
    np.testing.assert_allclose(edges, [-POINT_HALF_WIDTH, 0.0, POINT_HALF_WIDTH])


def test_common_edges_rejects_nothing_to_bin() -> None:
    with pytest.raises(ValueError, match="bins"):
        common_edges([np.ones(3)], bins=0)
    with pytest.raises(ValueError, match="No value"):
        common_edges([np.array([])])


def test_draw_histograms_overlays_densities_and_kdes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    samples = {"n = 100": rng.normal(size=100), "n = 2000": rng.normal(size=2000)}
    fig, ax = subplots()
    handles = draw_histograms(ax, samples, color_map(samples), bins=20)
    assert [h.get_label() for h in handles] == list(samples)
    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    assert len(bars) == 2
    for container in bars:
        widths = [p.get_width() for p in container.patches]
        heights = [p.get_height() for p in container.patches]
        assert np.dot(widths, heights) == pytest.approx(1.0)  # density
    # Both histograms share the same bins.
    assert [p.get_x() for p in bars[0].patches] == [p.get_x() for p in bars[1].patches]
    kde = handles[1]
    assert np.trapezoid(kde.get_ydata(), kde.get_xdata()) == pytest.approx(1, abs=0.05)
    data = json.loads(save_figure(fig, tmp_path, "h").with_suffix(".json").read_text())
    panel = data["panels"][0]
    assert [h["label"] for h in panel["histograms"]] == [
        "n = 100 (histogram)",
        "n = 2000 (histogram)",
    ]
    assert [c["label"] for c in panel["curves"]] == list(samples)


def test_a_constant_sample_gets_a_bar_but_no_kde() -> None:
    fig, ax = subplots()
    handles = draw_histograms(ax, {"<X>": np.zeros(50)}, {"<X>": "C0"}, bins=4)
    assert ax.get_lines() == []
    assert handles[0].get_label() == "<X>"
    assert len(ax.containers) == 1
