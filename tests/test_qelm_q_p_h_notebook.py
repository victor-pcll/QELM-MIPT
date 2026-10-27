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
        # The finite-size scaling section (#183): two sizes, two fields,
        # p = 0, 0.5 and 1, two runs at once.
        "FSS_N_VALUES": "3 4",
        "FSS_H_VALUES": "1 2",
        "FSS_P_POINTS": "3",
        "FSS_SEEDS": "42",
        "FSS_DELTA": "3",  # L = 9 at N = 3, 12 at N = 4
        "FSS_JOBS": "2",
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
    # The finite-size scaling (#183): its own sweep of
    # 2 N x 2 h x 3 p x 1 seed = 12 runs with the state vector, in its own
    # folder, one figure with a row per field, a column per score and a
    # curve per N, for the readout [M, <Z_k>] only.
    assert "12/12" in out
    assert namespace["FSS_P_VALUES"] == "0 0.5 1"
    assert namespace["fss_dir"].parent == tmp_path / "QELM_Q_N_scaling"
    fss_rows = namespace["fss_rows"]
    assert len(fss_rows) == 12
    for run in namespace["fss_dir"].glob("*/resolved_config.json"):
        assert json.loads(run.read_text())["state_representation"] == "vector"
    assert sorted(
        (r["N"], r["h"], r["p"], r["n_runs"]) for r in namespace["fss_table"]
    ) == [(N, h, p, 1) for N in (3, 4) for h in (1, 2) for p in (0.0, 0.5, 1.0)]
    axes_fss = namespace["axes_fss"]
    assert axes_fss.shape == (2, 2)
    for ax in axes_fss.flat:
        assert len(ax.containers) == 2
        assert "[M, <Z_k>]" in ax.get_title()
    assert "h = 2 J" in axes_fss[1, 0].get_title()
    labels = axes_fss[0, 0].get_legend_handles_labels()[1]
    assert sorted(labels) == ["N = 3, L = 9", "N = 4, L = 12", "chance"]
    assert "delta = 3" in axes_fss[0, 1].get_title()
    # L = int(delta N) in every run (#201).
    assert sorted({(r["N"], r["n_samples"]) for r in fss_rows}) == [(3, 9), (4, 12)]
    assert {r["delta"] for r in fss_rows} == {3.0}
    fss_path = namespace["figure_fss"]
    assert fss_path.parent == namespace["fss_dir"]
    data = json.loads(fss_path.with_suffix(".json").read_text())
    assert len(data["extra"]["table"]) == 12
    # The residuals 1 - score on a log scale, from the same table (no new run).
    axes_res = namespace["axes_res"]
    assert axes_res.shape == (2, 2)
    for ax in axes_res.flat:
        assert ax.get_yscale() == "log"
        assert "1 - " in ax.get_title()
    res_path = namespace["figure_res"]
    assert res_path == namespace["fss_dir"] / "residuals_vs_p_N.png"
    assert res_path.with_suffix(".json").is_file()
    # The Binder ratio of the prediction, refitted from the saved features.
    binder_table = namespace["binder_table"]
    assert len(binder_table) == 12
    assert all("binder_M_Z_mean" in r for r in binder_table)
    axes_binder = namespace["axes_binder"]
    assert axes_binder.shape == (2, 1)
    for ax in axes_binder.flat:
        assert len(ax.containers) == 2
        assert "Binder ratio" in ax.get_title()
    binder_path = namespace["figure_binder"]
    assert binder_path == namespace["fss_dir"] / "binder_vs_p_N.png"
    assert binder_path.with_suffix(".json").is_file()
    # p_linear*, the critical p of the ridge readout (#227): one row per
    # (N, h) and estimator whose curve is not all nan.
    estimates = namespace["p_linear_estimates"]
    assert {(e["N"], e["h"]) for e in estimates} == {(3, 1), (3, 2), (4, 1), (4, 2)}
    assert {"accuracy", "residual"} <= {e["estimator"] for e in estimates}
    assert all(0.0 <= e["p_refined"] <= 1.0 for e in estimates)
    assert namespace["axes_p_linear"].shape == (2, 2)
    assert namespace["axes_p_linear_N"].shape == (2,)
    for name in ("figure_p_linear", "figure_p_linear_N"):
        path = namespace[name]
        assert path.parent == namespace["fss_dir"]
        data = json.loads(path.with_suffix(".json").read_text())
        assert data["extra"]["estimates"] == json.loads(json.dumps(estimates))
    assert len(shows) == 11
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
