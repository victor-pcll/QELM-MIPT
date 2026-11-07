"""notebooks/results/12_fourier_expressivity.ipynb runs end to end (#309)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "12_fourier_expressivity.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "MODELS": "haar u1",
        "SIZES": "3:2 4:1",
        "P_VALUES": "0 0.5 1",
        "TASK_SAMPLES": "200",
        "DEGREES": "1 2",
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
    assert len(namespace["runs"]) == 6
    assert set(namespace["summary"]) == {"haar", "u1"}
    u1 = namespace["summary"]["u1"][4]
    assert max(u1["odd_x_weight_mean"]) < 1e-20
    assert len(shows) == 1
    sweep = namespace["figure_path"].parent
    assert sweep.parent == tmp_path / "fourier_expressivity"
    for name in ("accessible_rank", "final_mean_degree", "degree_profile_N4"):
        assert (sweep / f"{name}.png").is_file()
        assert (sweep / f"{name}.json").is_file()
    assert (sweep / "summary_table.json").is_file()
    assert (sweep / "task_table.json").is_file()
    for name in ("r2_qelm_k1", "accuracy_qelm_k2", "r2_ideal_k1", "r2_vs_shots_N4"):
        assert (sweep / f"{name}.png").is_file()
    assert {row["target"] for row in namespace["task_rows"]} == {"k1", "k2"}
