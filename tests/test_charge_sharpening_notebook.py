"""notebooks/benchmarks/03_charge_sharpening.ipynb runs end to end (#142).

The code cells are executed in order, as Jupyter would, on a small sweep set
through the environment variables the notebook reads.
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
NOTEBOOK = REPO_ROOT / "notebooks" / "benchmarks" / "03_charge_sharpening.ipynb"


# As in tests/test_qelm_m_notebook.py: show is a no-op under Agg, and the
# "non-interactive" warning is an error so that it cannot come back (#114).
@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "L": "30",
        "N_VALUES": "6",
        "SEEDS": "0 1",
        "GAMMA_VALUES": "0.0 0.5 1.0",
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
    # One bar for the sweep (6 runs), none for the trajectories of each run.
    shown = capsys.readouterr().out
    assert "6/6" in shown
    assert "30/30" not in shown
    assert len(namespace["rows"]) == 6
    table = namespace["table"]
    assert [(r["gamma"], r["n_runs"]) for r in table] == [
        (0.0, 2),
        (0.5, 2),
        (1.0, 2),
    ]
    # Exact limits: S_R = ln 2 without measurement, 0 with projective ones.
    assert table[0]["S_R_mean_mean"] == pytest.approx(np.log(2), abs=1e-9)
    assert table[-1]["S_R_mean_mean"] == pytest.approx(0.0, abs=1e-9)
    # The paper's 6-site curve, our N = 6 (issue #235), is compared; the
    # 20-site one (N = 20) is only drawn.
    comparison = namespace["comparison"]
    assert set(comparison["S_R_mean"]["N6"]) == {"N = 6"}
    assert comparison["S_R_mean"]["N6"]["N = 6"]["n_points"] > 0
    assert len(namespace["fig"].axes) == 3
    assert namespace["figure_path"].is_file()
    assert namespace["figure_path"].parent.parent == tmp_path / "charge_sharpening_fig7"
    assert (namespace["sweep_dir"] / "comparison.json").is_file()
    # Each figure is saved with the data it shows (#147).
    assert namespace["figure_path"].with_suffix(".json").is_file()
    assert namespace["hist_path"].parent == namespace["sweep_dir"]
    hist = json.loads(namespace["hist_path"].with_suffix(".json").read_text())
    assert len(hist["panels"]) == 3
    assert sum(hist["panels"][0]["histograms"][0]["heights"]) == 30
    # Histograms: one panel per gamma of the sweep (3 here, up to 5).
    assert namespace["hist_axes"].shape == (1, 3)
    assert len(shows) == 2  # Fig. 7 and the distributions
    os.chdir(REPO_ROOT)
