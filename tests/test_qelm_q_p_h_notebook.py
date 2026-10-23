"""notebooks/results/06_QELM_Q_observables_p_h.ipynb runs on a small sweep (#179, #183).

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
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "06_QELM_Q_observables_p_h.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    plt.close("all")  # figures left open by earlier tests
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N_VALUES": "3",
        "H_VALUES": "0 10",
        "P_VALUES": "0.0 1.0",
        "SEEDS": "42 43",
        "FIXED_P": "0.0 1.0",
        "L": "10",
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
        # The finite-size scaling section (#183): two sizes, p = 0 and 1.
        "FSS_N_VALUES": "3 4",
        "FSS_H": "1",
        "FSS_P_POINTS": "2",
        "FSS_SEEDS": "42",
        "FSS_L": "10",
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
    assert "8/8" in out  # one bar for the N x h x seed x p = 8 runs
    assert len(namespace["rows"]) == 8
    table = namespace["table"]
    assert sorted((r["h"], r["p"], r["n_runs"]) for r in table) == [
        (h, p, 2) for h in (0, 10) for p in (0.0, 1.0)
    ]
    for r in table:
        for key in ("accuracy_Z", "accuracy_XX"):
            assert 0.0 <= r[f"{key}_mean"] <= 1.0
    # One panel per observable and score, one curve per h (against p) or
    # per fixed p (against h), plus the chance line.
    assert namespace["axes_p"].shape == (1, 4)
    assert namespace["axes_h"].shape == (1, 4)
    for ax in namespace["axes_p"].flat:
        assert len(ax.containers) == 2
    for ax in namespace["axes_h"].flat:
        assert len(ax.containers) == 2
    # The heatmaps: one row per observable and score, one column per N, with
    # p on the x axis and h on the y axis.
    assert namespace["axes_maps"].shape == (4, 1)
    grid = namespace["grids"][(3, "XX", "r2")]
    np.testing.assert_array_equal(grid.x, [0.0, 1.0])
    np.testing.assert_array_equal(grid.y, [0.0, 10.0])
    assert np.all(grid.n_runs == 2)
    # The same figures for the readouts on the record: three readouts (M,
    # [M, <Z_k>], [M, <X_i X_j>]) of two scores each.
    assert namespace["record_axes_p"].shape == (1, 6)
    assert namespace["record_axes_h"].shape == (1, 6)
    assert namespace["record_axes_maps"].shape == (6, 1)
    # At p = 0 the record is all zeros: [M, X] scores as X.
    for r in table:
        if r["p"] == 0.0:
            for obs in ("Z", "XX"):
                assert r[f"accuracy_M_{obs}_mean"] == r[f"accuracy_{obs}_mean"]
    # The finite-size scaling at fixed h (#183): its own sweep of
    # 2 N x 2 p x 1 seed = 4 runs, in its own folder, one figure with a row
    # per readout (<Z_k>, <X_i X_j>, [M, <Z_k>]), a column per score and a
    # curve per N.
    assert "4/4" in out
    assert namespace["FSS_P_VALUES"] == "0 1"
    assert namespace["fss_dir"].parent == tmp_path / "QELM_Q_N_scaling"
    assert len(namespace["fss_rows"]) == 4
    assert {r["h"] for r in namespace["fss_rows"]} == {1.0}
    assert sorted((r["N"], r["p"], r["n_runs"]) for r in namespace["fss_table"]) == [
        (N, p, 1) for N in (3, 4) for p in (0.0, 1.0)
    ]
    assert namespace["axes_fss"].shape == (3, 2)
    for ax in namespace["axes_fss"].flat:
        assert len(ax.containers) == 2
    labels = namespace["axes_fss"][0, 0].get_legend_handles_labels()[1]
    assert sorted(labels) == ["N = 3", "N = 4", "chance"]
    fss_path = namespace["figure_fss"]
    assert fss_path.parent == namespace["fss_dir"]
    assert (
        len(json.loads(fss_path.with_suffix(".json").read_text())["extra"]["table"])
        == 4
    )
    assert len(shows) == 7
    # Every figure in this notebook's own folder (#151), with its data.
    for name in (
        "figure_p",
        "figure_h",
        "heatmap_path",
        "record_p",
        "record_h",
        "record_maps",
    ):
        path = namespace[name]
        assert path.parent.parent == tmp_path / "QELM_Q_p_h"
        data = json.loads(path.with_suffix(".json").read_text())
        assert len(data["extra"]["table"]) == 4
    # plt.show is patched out, so the figures stay open: close them.
    plt.close("all")
    os.chdir(REPO_ROOT)
