"""The benchmark overlays reproduce the paper curves exactly (#149).

Non-regression for the plot refactor of #148: the runs of a fake sweep sit
on the paper's 6-site curves plus ``DELTA`` (see ``plot_fixtures.py``,
which reads the CSV files on its own). The plot scripts must find a
deviation of exactly ``DELTA`` at every reference point, draw the reference
curves at the values read by hand, draw our curve at the values of the runs,
and keep the axis windows of the figures. The number of compared points and
the deviations were checked identical with the plot scripts of main before
#148 (aecb170).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot_fixtures import (
    DELTA,
    REPO_ROOT,
    charge_sharpening_reference,
    fake_charge_sharpening_sweep,
    fake_learnability_sweep,
    learnability_reference,
)
from QELM.utils.plots import figure_data

matplotlib.use("Agg")


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEARN = _load(REPO_ROOT / "experiments" / "learnability" / "plot_learnability.py")
CS = _load(
    REPO_ROOT / "experiments" / "charge_sharpening" / "plot_charge_sharpening.py"
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _curve(panel: dict, label: str) -> dict:
    (curve,) = [c for c in panel["curves"] if c["label"] == label]
    return curve


def _assert_deviation_is_delta(stats: dict, n_points: int) -> None:
    assert stats["n_points"] == n_points
    for key in ("max_abs_diff", "rms_diff", "mean_diff"):
        assert stats[key] == pytest.approx(DELTA, abs=1e-12), key


@pytest.mark.parametrize(
    ("figure", "metric", "n_points", "binder_xlim"),
    [("2b", "accuracy", 25, (0.0, 0.2)), ("2c", "entropy_mean", 24, (0.0, 0.4))],
)
def test_learnability_overlay_matches_the_paper(
    tmp_path: Path, figure: str, metric: str, n_points: int, binder_xlim: tuple
) -> None:
    sweep = fake_learnability_sweep(tmp_path / "sweep")
    panels, transition, _, _ = LEARN.FIGURES[figure]
    fig, comparison = LEARN.draw_sweep(sweep, LEARN.REFERENCE_DIR, panels, transition)

    # Our N = 6 is compared with the paper's 6-site curve only.
    assert set(comparison[metric]) == {"N6"}
    assert set(comparison[metric]["N6"]) == {"N = 6"}
    _assert_deviation_is_delta(comparison[metric]["N6"]["N = 6"], n_points)

    main, binder = figure_data(fig)["panels"]
    x_ref, y_ref = learnability_reference()[metric]
    paper = _curve(main, "Barratt et al. 2022: N = 6")
    np.testing.assert_allclose(paper["x"], x_ref, atol=1e-12)
    np.testing.assert_allclose(paper["y"], y_ref, atol=1e-12)
    ours = _curve(main, "this work, N = 6")
    np.testing.assert_allclose(
        ours["y"], np.interp(ours["x"], x_ref, y_ref) + DELTA, atol=1e-12
    )
    assert "Barratt et al. 2022: N = 20" in [c["label"] for c in main["curves"]]
    # The Binder inset keeps the window 0 <= p <= 2 p_c.
    assert fig.axes[1].get_xlim() == pytest.approx(binder_xlim)
    assert binder["title"]


def test_accuracy_axis_reads_0564_to_1(tmp_path: Path) -> None:
    """The Fig. 2b accuracy is drawn on the paper's scale, from 0.59 to 1."""
    sweep = fake_learnability_sweep(tmp_path / "sweep")
    panels, transition, _, _ = LEARN.FIGURES["2b"]
    fig, _ = LEARN.draw_sweep(sweep, LEARN.REFERENCE_DIR, panels, transition)
    main = figure_data(fig)["panels"][0]
    for name in ("N = 6", "N = 20"):
        y = np.array(_curve(main, f"Barratt et al. 2022: {name}")["y"])
        assert 0.58 < y.min() < 0.6 and 0.99 < y.max() <= 1.0 + 1e-9, name
    x = np.array(_curve(main, "Barratt et al. 2022: N = 6")["x"])
    assert 0.0 < x.min() < 0.05 and 0.45 < x.max() < 0.55


def test_charge_sharpening_overlay_matches_the_paper(tmp_path: Path) -> None:
    sweep = fake_charge_sharpening_sweep(tmp_path / "sweep")
    fig, comparison = CS.draw_sweep(sweep, CS.REFERENCE_DIR)

    refs = charge_sharpening_reference()
    panels = figure_data(fig)["panels"]
    for (metric, n_points), panel in zip(
        (("S_R_mean", 18), ("S_R_var", 18), ("binder", 12)), panels, strict=True
    ):
        # Our N = 7 (6 sites + the ancilla) against the paper's 6-site curve.
        assert set(comparison[metric]) == {"N7"}
        assert set(comparison[metric]["N7"]) == {"N = 7"}
        _assert_deviation_is_delta(comparison[metric]["N7"]["N = 7"], n_points)
        x_ref, y_ref = refs[metric]
        paper = _curve(panel, "Agrawal et al. 2024: N = 7")
        np.testing.assert_allclose(paper["x"], x_ref, atol=1e-12)
        np.testing.assert_allclose(paper["y"], y_ref, atol=1e-12)
        ours = _curve(panel, "this work, N = 7")
        np.testing.assert_allclose(
            ours["y"], np.interp(ours["x"], x_ref, y_ref) + DELTA, atol=1e-12
        )
    # The Binder panel keeps the scale of the reference curves.
    low = min(
        float(np.min(y))
        for name, (_, y) in CS.load_references(CS.REFERENCE_DIR)["binder"].items()
    )
    assert fig.axes[2].get_ylim() == pytest.approx((1.5 * low, 1.0))
