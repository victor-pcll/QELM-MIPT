"""Shared plotting helpers of QELM.utils.plots (issue #147)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from QELM.utils import summarize
from QELM.utils.plots import (
    STYLE,
    color_map,
    compare_same_size,
    draw_curve,
    draw_references,
    figure_data,
    print_comparison,
    save_figure,
    sizes,
    style_axes,
    subplots,
    sweep_curve,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rows() -> list[dict]:
    """Two sizes, two values of p; N = 4 has two seeds, N = 6 one."""
    rows = []
    for seed, shift in ((0, 0.0), (1, 0.2)):
        for p in (0.5, 0.1):
            rows.append({"N": 4, "p": p, "acc": p + shift, "acc_err": 0.05})
    for p in (0.1, 0.5):
        rows.append({"N": 6, "p": p, "acc": 2 * p, "acc_err": 0.03})
    return rows


def _summary() -> list[dict]:
    return summarize(_rows(), by=("N", "p"), metrics=["acc", "acc_err"])


def test_subplots_has_no_grid_a_closed_box_and_large_titles() -> None:
    fig, axes = subplots(1, 2)
    for ax in axes:
        assert not any(line.get_visible() for line in ax.get_xgridlines())
        assert not any(line.get_visible() for line in ax.get_ygridlines())
        assert all(spine.get_visible() for spine in ax.spines.values())
        assert set(ax.spines) == {"top", "right", "bottom", "left"}
    title = axes[0].set_title("A title")
    assert title.get_fontsize() == STYLE["axes.titlesize"] >= 14
    assert title.get_fontweight() == "bold"
    assert fig.suptitle("Figure").get_fontsize() == STYLE["figure.titlesize"]
    assert axes[0].xaxis.label.get_fontsize() == STYLE["axes.labelsize"]


def test_style_axes_fixes_axes_made_elsewhere() -> None:
    with plt.rc_context({"axes.grid": True, "axes.spines.top": False}):
        fig, ax = plt.subplots()
    ax.spines[["right"]].set_visible(False)
    style_axes(ax)
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert all(spine.get_visible() for spine in ax.spines.values())


def test_color_map_is_ordered_and_cycles() -> None:
    colors = color_map([f"k{k}" for k in range(8)])
    assert colors["k0"] != colors["k1"]
    assert colors["k0"] == colors["k6"]


def test_sweep_curve_orders_by_x_and_picks_the_error() -> None:
    summary = _summary()
    assert sizes(summary) == [4, 6]
    four = sweep_curve(summary, 4, "p", "acc")
    np.testing.assert_allclose(four.x, [0.1, 0.5])
    np.testing.assert_allclose(four.mean, [0.2, 0.6])
    # Two seeds: the standard error over the seeds, 0.1 here.
    np.testing.assert_allclose(four.err, [0.1, 0.1])
    six = sweep_curve(summary, 6, "p", "acc")
    # One seed: the run's own error (acc_err).
    np.testing.assert_allclose(six.err, [0.03, 0.03])
    kept = six.where(np.array([False, True]))
    assert kept.x.tolist() == [0.5] and len(kept.rows) == 1


def test_compare_same_size_skips_other_sizes() -> None:
    curve = sweep_curve(_summary(), 6, "p", "acc")
    refs = {
        "N = 6": (np.array([0.1, 0.5]), np.array([0.2, 1.0])),
        "N = 20": (np.array([0.1]), np.array([0.0])),
        "unsized": (np.array([0.3]), np.array([0.6])),
    }
    stats = compare_same_size(curve, 6, refs)
    assert set(stats) == {"N = 6", "unsized"}
    assert stats["N = 6"]["max_abs_diff"] == pytest.approx(0.0)
    assert stats["unsized"]["n_points"] == 1


def test_save_figure_writes_the_png_and_its_data(tmp_path: Path) -> None:
    fig, axes = subplots(1, 2)
    curve = sweep_curve(_summary(), 4, "p", "acc")
    draw_curve(axes[0], curve, "#000000", label="N = 4", band=True)
    draw_references(
        axes[0],
        {"N = 4": (np.array([0.0, 1.0]), np.array([0.0, 1.0]))},
        {"N = 4": "#ff0000"},
        "Paper",
    )
    axes[0].set(title="Accuracy", xlabel="p", ylabel="acc")
    axes[1].hist([0.1, 0.2, 0.2, 0.9], bins=[0.0, 0.5, 1.0], label="hist")
    fig.suptitle("Test")

    png = save_figure(fig, tmp_path / "sweep", "fig", data={"n": np.int64(3)})
    assert png == tmp_path / "sweep" / "fig.png" and png.is_file()
    data = json.loads((tmp_path / "sweep" / "fig.json").read_text())
    assert data == {**figure_data(fig), "extra": {"n": 3}}
    assert data["title"] == "Test"
    left, right = data["panels"]
    assert left["title"] == "Accuracy" and left["xlabel"] == "p"
    ours = next(c for c in left["curves"] if c["label"] == "N = 4")
    assert ours["x"] == [0.1, 0.5]
    np.testing.assert_allclose(ours["y"], [0.2, 0.6])
    np.testing.assert_allclose(ours["yerr_lower"], [0.1, 0.1])
    np.testing.assert_allclose(ours["yerr_upper"], [0.1, 0.1])
    ref = next(c for c in left["curves"] if c["label"] == "Paper: N = 4")
    assert ref["y"] == [0.0, 1.0]
    (hist,) = right["histograms"]
    assert hist["edges"] == [0.0, 0.5, 1.0] and hist["heights"] == [3.0, 1.0]


def test_log_curve_keeps_its_lower_bar_above_zero(tmp_path: Path) -> None:
    fig, ax = subplots()
    curve = sweep_curve(_summary(), 4, "p", "acc")
    draw_curve(ax, curve, log=True)
    assert ax.get_yscale() == "log"
    save_figure(fig, tmp_path, "log")
    (panel,) = json.loads((tmp_path / "log.json").read_text())["panels"]
    (drawn,) = panel["curves"]
    # Lower bar stopped at half the mean: 0.1 at 0.2 becomes 0.1, 0.1 at
    # 0.6 stays 0.1, so check the bound only.
    y, low = np.array(drawn["y"]), np.array(drawn["yerr_lower"])
    assert np.all(y - low > 0)


@pytest.mark.parametrize("name", ["", "a/b", "fig.png"])
def test_save_figure_rejects_paths_and_suffixes(tmp_path: Path, name: str) -> None:
    fig, _ = subplots()
    with pytest.raises(ValueError, match="plain file name"):
        save_figure(fig, tmp_path, name)


def test_print_comparison(capsys: pytest.CaptureFixture[str]) -> None:
    stats = {"max_abs_diff": 0.01, "rms_diff": 0.005, "n_points": 4}
    print_comparison(
        {
            "accuracy": {"N6": {"N = 6": stats}},
            "binder_crossings": {"binder": {"N4-N6": 0.12, "N6-N8": None}},
        }
    )
    out = capsys.readouterr().out
    assert "binder crossing N4-N6: at 0.120" in out
    assert "binder crossing N6-N8: no crossing" in out
    assert "accuracy N6 vs N = 6: max |diff| = 0.01" in out


def test_importing_QELM_does_not_load_matplotlib() -> None:
    """Matplotlib is a dev dependency: QELM.utils.plots is imported apart."""
    code = "import sys, QELM, QELM.utils; print('matplotlib' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
