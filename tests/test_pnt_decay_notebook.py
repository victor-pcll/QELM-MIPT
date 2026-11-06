"""notebooks/results/11_pnt_decay.ipynb runs end to end (#305)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "11_pnt_decay.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "ENCODINGS": "charge",
        "SIZES": "3:2,4 4:2",
        "SEEDS": "1 2",
        "PER_P": "60",
        "PNT_VALUES": "0 1 3 6",
        "MAX_JOBS": "2",
        "OUT_ROOT": str(tmp_path),
        "RUN": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOK.parent)

    namespace: dict = {}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)

    assert Path.cwd() == REPO_ROOT
    assert namespace["n_runs"] == 6
    assert len(namespace["rows"]) == 6
    assert set(namespace["summary"]["encodings"]) == {"charge"}
    assert len(shows) == 7
    sweep = namespace["figure_path"].parent
    assert sweep.parent == tmp_path / "pnt_decay"
    for name in ("signal_decay", "accuracy_vs_snr", "midpoints"):
        assert (sweep / f"{name}.png").is_file()
