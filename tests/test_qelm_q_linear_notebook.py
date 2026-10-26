"""notebooks/results/08_QELM_Q_linear_observables.ipynb on a small run (#191).

The code cells are executed in order, as Jupyter would, with the datasets
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
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "08_QELM_Q_linear_observables.ipynb"


def _run_cells(namespace: dict) -> None:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    plt.close("all")  # figures left open by earlier tests
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N": "5",
        "H_VALUES": "10 1",
        "N_SAMPLES": "10 20",
        "BINS": "5",
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOK.parent)  # where Jupyter starts

    namespace: dict = {}
    _run_cells(namespace)

    assert Path.cwd() == REPO_ROOT
    runs = namespace["runs"]
    assert sorted((r["h"], r["n_samples"]) for r in runs) == [
        (1.0, 10),
        (1.0, 20),
        (10.0, 10),
        (10.0, 20),
    ]
    # Independent datasets: one seed per size.
    assert {r["n_samples"]: r["seed"] for r in runs} == {10: 42, 20: 43}
    for r in runs:
        assert set(r["obs"]) == {"Z", "Y", "X"}
        assert r["obs"]["Z"].shape == (r["n_samples"], 4)
        assert set(np.unique(r["Y"])) <= {2, 1}  # Q1 = Ns // 2, Q2 = Q1 - 1
        # Parity: no linear X or Y feature.
        np.testing.assert_allclose(r["obs"]["X"], 0.0, atol=1e-10)
        np.testing.assert_allclose(r["obs"]["Y"], 0.0, atol=1e-10)
    assert len(namespace["ridge_table"]) == 4
    # One row per field; columns Q1, Q2 and both, one histogram per size,
    # one Gaussian fit each, no KDE and no Haar curve (PR #192).
    axes_h = namespace["axes_h"]
    assert axes_h.shape == (2, 3)
    for ax in axes_h.flat:
        assert len(ax.containers) == 2
        labels = [line.get_label() for line in ax.get_lines()]
        assert len(labels) == 1 and labels[0].startswith("Gaussian")
    assert "Q = 2" in axes_h[0, 0].get_title()
    assert "Q = 1" in axes_h[0, 1].get_title()
    assert "Q1 and Q2" in axes_h[0, 2].get_title()
    # The two sectors split each dataset.
    r = runs[0]
    n_q1 = len(namespace["sector_values"](r, "Q = 2"))
    n_q2 = len(namespace["sector_values"](r, "Q = 1"))
    assert n_q1 + n_q2 == r["n_samples"]
    assert len(namespace["convergence"]) == 2 * 2
    assert len(namespace["shape_table"]) == 2 * 3 * 4
    assert len(shows) == 2

    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "QELM_Q_linear"
    assert sorted(p.name for p in sweep_dir.iterdir() if p.is_dir()) == [
        "h10_n10",
        "h10_n20",
        "h1_n10",
        "h1_n20",
    ]
    for name in (
        "ridge_vs_n_samples",
        "histograms_linear_observables",
        "convergence_vs_n_samples",
    ):
        assert (sweep_dir / f"{name}.png").is_file()
        assert json.loads((sweep_dir / f"{name}.json").read_text())["panels"]
    histograms = json.loads(
        (sweep_dir / "histograms_linear_observables.json").read_text()
    )
    assert len(histograms["extra"]["shape"]) == 24
    plt.close("all")

    # RUN = False reads the same sweep back without simulating.
    monkeypatch.setenv("RUN", "0")
    again: dict = {}
    _run_cells(again)
    assert again["sweep_dir"] == sweep_dir
    np.testing.assert_array_equal(again["runs"][0]["obs"]["Z"], runs[0]["obs"]["Z"])
    plt.close("all")
    os.chdir(REPO_ROOT)
