"""notebooks/results/02_QELM_M_analysis.ipynb runs end to end on a small sweep (#94).

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
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "02_QELM_M_analysis.ipynb"


# Agg has no window: plt.show() in the notebook warns "FigureCanvasAgg is
# non-interactive" whenever a display is set (macOS, Linux with DISPLAY).
# show is replaced by a no-op below, and the warning is an error here so that
# it cannot come back (#114).
@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "L": "10",
        "N_VALUES": "3",
        "SEEDS": "42 43",
        "P_VALUES": "0.2 1.0",
        "SCALING_N_VALUES": "3",
        "H_VALUES": "1 10",
        "SCALING_P_VALUES": "0.2 1.0",
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
    # One bar per sweep: 4 runs for the first one, 2 x 2 x 2 for the sweep
    # over h (#174); none for the samples of each run.
    shown = capsys.readouterr().out
    assert "4/4" in shown
    assert "8/8" in shown
    assert "10/10" not in shown
    assert len(namespace["rows"]) == 4
    assert [(r["p"], r["n_runs"]) for r in namespace["table"]] == [(0.2, 2), (1.0, 2)]
    # Every metric of the runs is printed in the table, the redundant ones
    # included (#174).
    keys = [key for key, _ in namespace["METRICS"]]
    assert keys == [
        "accuracy",
        "accuracy_baseline",
        "accuracy_optimal",
        "balanced_accuracy",
        "r2",
        "nmse",
        "S_vN_mean",
        "S_renyi_1_mean",
        "S_vN_sys_mean",
        "S_renyi_2_sys_mean",
        "S_half_mean",
        "S_vN_1q_mean",
        "S_renyi_2_1q_mean",
    ]
    for key in keys:
        assert all(np.isfinite(r[f"{key}_sem"]) for r in namespace["table"])
    # Two figures, one row each (#174): performance (accuracy, R2) and
    # entanglement (ancilla, half-system, single-qubit). The redundant
    # metrics (NMSE, S_renyi_1, S_vN_sys, balanced accuracy) are not drawn.
    perf, ent = (
        namespace["axes_of"]["performance_vs_p"],
        namespace["axes_of"]["entanglement_vs_p"],
    )
    assert [ax.get_title() for ax in perf] == [
        "Accuracy of label Y",
        "R2 of the prediction",
    ]
    assert [ax.get_title() for ax in ent] == [
        "Ancilla entropy",
        "Half-system entropy",
        "Mean single-qubit entropy",
    ]
    # The accuracy panel carries the chance level, dashed, and the Bayes
    # ceiling, dotted, in the color of the same N, with their own error bars.
    containers = {c.get_label(): c for c in perf[0].containers}
    assert set(containers) == {
        "N = 3",
        "N = 3, chance (majority sign)",
        "N = 3, Bayes ceiling",
    }
    color = containers["N = 3"].lines[0].get_color()
    for label, style in (
        ("N = 3, chance (majority sign)", "--"),
        ("N = 3, Bayes ceiling", ":"),
    ):
        assert containers[label].lines[0].get_linestyle() == style
        assert containers[label].lines[0].get_color() == color
    np.testing.assert_allclose(
        containers["N = 3, Bayes ceiling"].lines[0].get_ydata(),
        [r["accuracy_optimal_mean"] for r in namespace["table"]],
    )
    # The ancilla panel: S_vN, its Rényi-2 dashed, and the 1 bit bound.
    labels = [c.get_label() for c in ent[0].containers]
    assert labels == ["N = 3", "N = 3, Rényi-2"]
    assert any(
        line.get_linestyle() == ":" and set(line.get_ydata()) == {1.0}
        for line in ent[0].lines
    )
    # p = 1 purifies the ancilla and every system qubit in every run, so the
    # entropies and their errors vanish there.
    at_p1 = next(r for r in namespace["table"] if r["p"] == 1.0)
    for key in ("S_vN_mean", "S_half_mean"):
        assert at_p1[f"{key}_mean"] == pytest.approx(0.0, abs=1e-9)
        assert at_p1[f"{key}_sem"] == pytest.approx(0.0, abs=1e-9)
    figures = namespace["figures"]
    assert set(figures) == {"performance_vs_p", "entanglement_vs_p"}
    for name, path in figures.items():
        assert path.is_file()
        assert path.parent.parent == tmp_path / "QELM_M"  # #151
        # The plotted data is saved next to the figure (#147).
        data = json.loads(path.with_suffix(".json").read_text())
        assert len(data["panels"]) == len(namespace["axes_of"][name])
        assert len(data["extra"]["table"]) == 2
    # The first sweep keeps the field of the config, h = 10 J, even with
    # H_VALUES set for the last section.
    assert {r["run"] for r in namespace["rows"]} == {
        "N3_seed42_p0.2",
        "N3_seed43_p0.2",
        "N3_seed42_p1.0",
        "N3_seed43_p1.0",
    }

    # The sweep over h and N (#174): its own folder, one row per (N, h, p).
    assert namespace["scaling_dir"].parent == tmp_path / "QELM_M_h"
    assert len(namespace["scaling_rows"]) == 8
    assert sorted({r["h"] for r in namespace["scaling_rows"]}) == [1.0, 10.0]
    assert [(r["h"], r["p"], r["n_runs"]) for r in namespace["scaling_table"]] == [
        (1.0, 0.2, 2),
        (1.0, 1.0, 2),
        (10.0, 0.2, 2),
        (10.0, 1.0, 2),
    ]
    axes_s = namespace["axes_s"][0]
    assert [ax.get_title() for ax in axes_s] == [
        "Accuracy of label Y",
        "R2 of the prediction",
        "Half-system entropy",
    ]
    for ax in axes_s:
        styles = {c.get_label(): c.lines[0].get_linestyle() for c in ax.containers}
        assert styles == {"N = 3, h = 10 J": "-", "N = 3, h = 1 J": "--"}
    assert namespace["scaling_figure"].is_file()
    data = json.loads(namespace["scaling_figure"].with_suffix(".json").read_text())
    assert len(data["panels"]) == 3
    assert len(data["extra"]["table"]) == 4
    assert len(shows) == 3  # the notebook asks to show its three figures
    os.chdir(REPO_ROOT)
