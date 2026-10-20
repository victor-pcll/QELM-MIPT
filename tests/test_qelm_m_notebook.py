"""notebooks/02_QELM_M_analysis.ipynb runs end to end on a small sweep (#94).

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
NOTEBOOK = REPO_ROOT / "notebooks" / "02_QELM_M_analysis.ipynb"


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
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(REPO_ROOT / "notebooks")  # where Jupyter starts

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)

    assert Path.cwd() == REPO_ROOT
    # One bar for the sweep (4 runs), none for the samples of each run.
    shown = capsys.readouterr().out
    assert "4/4" in shown
    assert "10/10" not in shown
    assert len(namespace["rows"]) == 4
    assert [(r["p"], r["n_runs"]) for r in namespace["table"]] == [(0.2, 2), (1.0, 2)]
    # R2, NMSE, the accuracy, the two ancilla entropies (#119) and the system
    # entropies (#124), each with a mean +- error plot and an error-only plot.
    keys = [key for key, _, _ in namespace["METRICS"]]
    assert keys == [
        "r2",
        "nmse",
        "accuracy",
        "S_vN_mean",
        "S_renyi_1_mean",
        "S_vN_sys_mean",
        "S_renyi_2_sys_mean",
        "S_vN_1q_mean",
        "S_renyi_2_1q_mean",
    ]
    assert namespace["axes"].shape == (len(keys), 2)
    for key in keys:
        assert all(np.isfinite(r[f"{key}_sem"]) for r in namespace["table"])
    # p = 1 purifies the ancilla in every run, so the entropy and its error
    # vanish there (S_vN_mean, the mean over the samples of a run, averaged
    # over the seeds by summarize).
    at_p1 = next(r for r in namespace["table"] if r["p"] == 1.0)
    assert at_p1["S_vN_mean_mean"] == pytest.approx(0.0, abs=1e-9)
    assert at_p1["S_vN_mean_sem"] == pytest.approx(0.0, abs=1e-9)
    assert namespace["figure_path"].is_file()
    assert len(shows) == 1  # the notebook still asks to show its figure
    assert namespace["figure_path"].parent.parent == tmp_path / "exp_QELM_M"
    os.chdir(REPO_ROOT)
