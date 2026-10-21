"""notebooks/results/03_QELM_M_alpha_ratio.ipynb runs end to end on a small sweep (#96).

The code cells are executed in order, as Jupyter would, with the sweep
shrunk through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
        "N_VALUES": "3",
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
    assert "8/8" in capsys.readouterr().out  # one bar for the 8 runs
    assert len(namespace["rows"]) == 8
    # N = 3: 11 weights, so the ratios 0.5 and 2.0 give 6 and 22 training
    # samples, i.e. 6/11 and 2 after rounding.
    table = namespace["table"]
    assert [(r["p"], r["ratio"], r["n_runs"]) for r in table] == [
        (0.5, pytest.approx(6 / 11), 2),
        (0.5, 2.0, 2),
        (1.0, pytest.approx(6 / 11), 2),
        (1.0, 2.0, 2),
    ]
    axes = namespace["axes"]
    assert len(axes) == len(namespace["METRICS"]) == 3
    for ax in axes:
        assert ax.get_xscale() == "log"
        # The interpolation threshold at alpha_ratio = 1.
        assert any(
            line.get_linestyle() == "--" and set(line.get_xdata()) == {1.0}
            for line in ax.lines
        )
    assert axes[0].get_yscale() == "log"  # NMSE
    assert len(shows) == 1
    figure_path = namespace["figure_path"]
    assert figure_path.parent.parent == tmp_path / "QELM_M_ratio"  # #151
    data = json.loads(figure_path.with_suffix(".json").read_text())
    assert len(data["panels"]) == 3
    assert len(data["extra"]["table"]) == 4
    os.chdir(REPO_ROOT)
