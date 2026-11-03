"""notebooks/results/09_kl_divergence.ipynb runs end to end (#270).

The code cells are executed in order, as Jupyter would, with the sweep
shrunk through the environment variables the notebook reads.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "09_kl_divergence.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "L": "40",
        "N_C": "2",
        "PER_CIRCUIT": "20",
        "N_VALUES": "4 6",
        "SEEDS": "0",
        "P_VALUES": "0.0 0.3",
        "JOBS": "2",
        "P_C": "0.15",
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
    table = namespace["table"]
    assert [(r["N"], r["p"]) for r in table] == [(4, 0.0), (4, 0.3), (6, 0.0), (6, 0.3)]
    for r in table:
        if r["p"] == 0.0:
            # No measurement: the two laws are the same.
            assert r["sep_bayes_accuracy_mean"] == 0.5
            assert r["sep_js_bits_mean"] == 0.0
            assert r["sep_kl_plus_minus_mean"] == pytest.approx(0.0, abs=1e-12)
        else:
            assert r["sep_bayes_accuracy_mean"] > 0.5
            assert r["circuit_js_bits_mean"] > 0.0
            assert math.isinf(r["sep_kl_plus_minus_mean"])
    sweep_dir = namespace["sweep_dir"]
    assert sweep_dir.parent == tmp_path / "kl_divergence_notebook"
    for name in ("kl_vs_accuracy", "kl_finite_size", "kl_bounds"):
        assert (sweep_dir / f"{name}.png").is_file()
        assert (sweep_dir / f"{name}.json").is_file()
    assert (sweep_dir / "scaling.json").is_file()
    assert len(shows) == 4
    os.chdir(REPO_ROOT)
