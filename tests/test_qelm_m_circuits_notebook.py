"""notebooks/results/05_QELM_M_circuits.ipynb runs end to end on a small sweep.

Issue #171: the ratio sweep of notebook 03 once per unitary dynamics (TFIM,
Haar and Clifford circuits), its curves, heatmaps and comparison figure.
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
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "05_QELM_M_circuits.ipynb"
MODELS = ("TFIM", "haar_circuit", "clifford_circuit")


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_three_sweeps_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "MODELS": " ".join(MODELS),
        "N_VALUES": "3",
        "SEEDS": "42 43",
        "P_VALUES": "0.0 1.0",
        "ALPHA_RATIOS": "0.5 2.0",
        "N_TEST": "10",
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
        "TQDM_DISABLE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOK.parent)  # where Jupyter starts

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)

    assert Path.cwd() == REPO_ROOT
    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "QELM_M_circuits"  # #151
    # One subfolder per model, 2 seeds x 2 p x 2 ratios runs each.
    tables = namespace["tables"]
    assert list(tables) == list(MODELS)
    for model in MODELS:
        assert len(list((sweep_dir / model).glob("N3_*"))) == 8
        table = tables[model]
        assert [(r["p"], r["n_runs"]) for r in table] == [(0.0, 2)] * 2 + [(1.0, 2)] * 2
        # p = 0: the record is empty, the Bayes classifier is at chance.
        for r in table:
            if r["p"] == 0.0:
                assert r["accuracy_optimal_mean"] == pytest.approx(0.5)
                assert r["S_vN_mean_mean"] == pytest.approx(1.0)

    # Curves and heatmaps: one row of panels per model and N.
    assert namespace["axes"].shape == (3, 3)
    assert namespace["axes_maps"].shape == (3, 4)
    for row, model in zip(namespace["axes"], MODELS, strict=True):
        for ax in row:
            assert ax.get_title().endswith(f"{model}, N = 3")
            assert ax.get_xscale() == "log"
    for model in MODELS:
        grid = namespace["grids"][(model, 3, "balanced_accuracy")]
        np.testing.assert_allclose(grid.x, [6 / 11, 2.0])
        np.testing.assert_array_equal(grid.y, [0.0, 1.0])
        assert np.all(np.isfinite(grid.mean))

    # The comparison: one curve per model in each panel.
    axes_cmp = namespace["axes_cmp"]
    assert axes_cmp.shape == (1, 3)
    for ax in axes_cmp[0]:
        labels = ax.get_legend_handles_labels()[1]
        assert [lab for lab in labels if lab in MODELS] == list(MODELS)
    assert axes_cmp[0, 2].get_yscale() == "log"
    assert len(shows) == 3
    for name in ("metrics_vs_alpha_ratio", "heatmaps_p_alpha_ratio"):
        assert (sweep_dir / f"{name}.png").is_file()
    data = json.loads((sweep_dir / "comparison_models.json").read_text())
    assert set(data["extra"]["comparison"]) == {f"{m} N=3" for m in MODELS}
    os.chdir(REPO_ROOT)
