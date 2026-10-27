"""notebooks/results/07_double_descent_mnist.ipynb runs end to end (#181, #196).

The code cells are executed in order, as Jupyter would, with the sweep shrunk
through the environment variables the notebook reads, on scikit-learn's 8x8
digits so that nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "results" / "07_double_descent_mnist.ipynb"


@pytest.mark.filterwarnings("error:.*FigureCanvasAgg is non-interactive:UserWarning")
def test_notebook_runs_the_sweep_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    shows: list[tuple] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(args))
    for key, value in {
        "N": "3",
        "SEEDS": "0 1",
        "DATASET": "digits",
        "N_POOL": "200",
        "N_REPEATS": "3",
        "CLASSES": "3 5",
        "N_CHECK": "20",
        "ENCODINGS": "angle_encoding dense_angle_encoding",
        "EXTRA_OVERRIDES": "double_descent.alpha_ratios=[0.5,1.0,2.0,4.0]",
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
    out = capsys.readouterr().out
    assert "2/2" in out  # one bar for the two seeds
    assert "t = 0 features vs closed form" in out
    assert len(namespace["runs"]) == 2
    table = namespace["table"]
    assert table["models"]["qelm"]["peak_alpha_of_mean"] == 1.0
    assert namespace["axes"].shape == (2, 2)
    # The double descent, the confusion matrices, then the encodings (#196).
    assert len(shows) == 3
    assert len(namespace["axes_confusion"]) == 5
    figure_path = namespace["figure_path"]
    assert figure_path.is_file()
    assert figure_path.parent.parent == tmp_path / "double_descent"  # #151
    assert (figure_path.parent / "summary.json").is_file()
    assert (figure_path.parent / "confusion_matrices.png").is_file()
    # Issue #196: the small test of the four encodings on a few images...
    check = namespace["encoding_check"]
    assert list(check) == [
        "angle_encoding",
        "dense_angle_encoding",
        "uniform_bloch_encoding",
        "general_encoding",
    ]
    for name, row in check.items():
        assert row["n_inputs"] == (3 if name == "angle_encoding" else 6)
        assert row["norm_deviation"] < 1e-12
        assert row["bloch_deviation"] < 1e-12
        assert row["out_of_bounds_raises"]
    assert check["angle_encoding"]["mean_abs_Y"] < 1e-12
    assert check["dense_angle_encoding"]["mean_abs_Y"] > 0.01
    # ...then the comparison, one sweep per encoding in its own folder.
    encodings_dir = namespace["encodings_dir"]
    assert encodings_dir.parent == tmp_path / "double_descent_encodings"
    assert sorted(p.name for p in encodings_dir.iterdir() if p.is_dir()) == [
        "angle_encoding",
        "dense_angle_encoding",
    ]
    assert (encodings_dir / "encodings.png").is_file()
    assert (encodings_dir / "encodings_summary.json").is_file()
    assert namespace["encodings_table"]["dense_angle_encoding"]["n_inputs"] == 6
    assert len(namespace["axes_encodings"]) == 2
