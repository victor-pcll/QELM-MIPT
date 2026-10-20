"""The learnability benchmark in experiments/learnability/ (issue #113)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "learnability"
SCRIPT = FOLDER / "learnability.py"
PLOT = FOLDER / "plot_learnability.py"
BASH_SCRIPT = FOLDER / "learnability.sh"
FOREIGN_VENV = REPO_ROOT.parent / "other-project" / ".venv"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*overrides: str):  # noqa: ANN202
    return load_config(
        [
            "samples.L=200",
            "learnability.n_boot=20",
            "learnability.batch=64",
            *overrides,
        ],
        config_name="learnability",
    )


def test_config_composes_with_the_paper_parameters() -> None:
    cfg = load_config(config_name="learnability")
    assert (cfg.N, cfg.Ns, cfg.Nr) == (6, 6, 0)
    assert cfg.samples.L == 40000
    assert cfg.learnability.L == 6
    assert cfg.learnability.depth == 6  # t_f = L
    assert cfg.learnability.boundary == "open"
    cfg = load_config(["N=8"], config_name="learnability")
    assert (cfg.learnability.L, cfg.learnability.depth) == (8, 8)


def test_run_shapes_and_ranges() -> None:
    out = _load(SCRIPT).run(_cfg("learnability.p=0.2"))
    assert out["P_corr"].shape == (200,)
    # t_f = 6 steps, measured after each of their 2 layers: 12 rounds.
    assert out["outcomes"].shape == (200, 12, 6)
    assert set(np.unique(out["outcomes"])) <= {-1, 0, 1}
    # Batches of 64 keep the charges alternating: half of each.
    assert sorted(np.unique(out["charge"], return_counts=True)[1]) == [100, 100]
    assert np.all((out["P_corr"] >= 0) & (out["P_corr"] <= 1))
    assert 0.5 <= out["accuracy"] <= 1
    assert out["measured_fraction"] == pytest.approx(0.2, abs=0.03)
    for key in _load(SCRIPT).STATISTICS:
        assert np.isfinite(out[key]) or "binder" in key


def test_no_measurement_is_a_blind_guess() -> None:
    """p = 0 : aucun relevé, P_corr = 1/2 partout, accuracy = 1/2, H = 1 bit."""
    out = _load(SCRIPT).run(_cfg("learnability.p=0.0"))
    np.testing.assert_allclose(out["P_corr"], 0.5)
    assert out["accuracy"] == 0.5
    assert out["accuracy_strict"] == 0.0
    assert out["entropy_mean"] == pytest.approx(np.log(2))
    assert out["measured_fraction"] == 0.0


def test_full_measurement_reveals_the_charge() -> None:
    """p = 1 : la charge se lit sur le premier relevé, accuracy = 1."""
    out = _load(SCRIPT).run(_cfg("learnability.p=1.0"))
    np.testing.assert_allclose(out["P_corr"], 1.0)
    assert out["accuracy"] == 1.0
    np.testing.assert_array_equal(out["outcomes"][:, 0].sum(axis=1), out["charge"])


def test_accuracy_grows_with_p() -> None:
    run = _load(SCRIPT).run
    acc = [
        run(_cfg(f"learnability.p={p}", "samples.L=2000", "learnability.batch=1000"))[
            "accuracy"
        ]
        for p in (0.05, 0.2, 0.5)
    ]
    assert acc[0] < acc[1] < acc[2]


def test_seed_reproduces_and_batch_must_be_even() -> None:
    run = _load(SCRIPT).run
    a = run(_cfg("learnability.p=0.3", "seed=3"))
    b = run(_cfg("learnability.p=0.3", "seed=3"))
    np.testing.assert_array_equal(a["P_corr"], b["P_corr"])
    c = run(_cfg("learnability.p=0.3", "seed=4"))
    assert not np.array_equal(a["P_corr"], c["P_corr"])
    with pytest.raises(ValueError):
        run(_cfg("learnability.batch=3"))


def test_curve_crossing() -> None:
    crossing = _load(PLOT).curve_crossing
    x = np.array([0.0, 0.1, 0.2, 0.3])
    assert crossing(x, x, 0.15 + 0 * x) == pytest.approx(0.15)
    assert crossing(x, x, x + 1) is None
    assert crossing(x, np.array([1.0, np.nan, 0.0, 0.0]), 0.5 + 0 * x) == (
        pytest.approx(0.1)
    )


def test_accuracy_lower_bound() -> None:
    """Eq. (1) : 1/2 à p = 0, ~0.84 à p = 0.5 (courbe en tirets de la Fig. 2b)."""
    bound = _load(PLOT).accuracy_lower_bound(np.array([0.0, 0.5]))
    np.testing.assert_allclose(bound, [0.5, 0.8413], atol=1e-4)


def test_bash_script_and_plot(tmp_path: Path) -> None:
    """Le lanceur fait 2 L x 1 seed x 3 p, puis la figure et la comparaison."""
    assert os.access(BASH_SCRIPT, os.X_OK) and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "40",
            "N_VALUES": "4 6",
            "P_VALUES": "0.0 0.3 1.0",
            "VIRTUAL_ENV": str(FOREIGN_VENV),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 2 x 1 x 3 runs" in result.stdout
    sweep = next((tmp_path / "learnability").iterdir())
    runs = sorted(p.name for p in sweep.iterdir() if p.is_dir())
    assert runs[0] == "N4_seed0_p0.0" and len(runs) == 6
    metrics = json.loads((sweep / "N6_seed0_p1.0" / "metrics.json").read_text())
    assert metrics["p"] == 1.0 and metrics["accuracy"] == 1.0
    assert metrics["L"] == 6 and metrics["n_records"] == 40
    assert (sweep / "learnability.png").is_file()
    assert (sweep / "learnability_fig2c.png").is_file()
    assert (sweep / "comparison_fig2c.json").is_file()

    # Against a fake reference that our two L = 6 points match, and one
    # for another size, which is drawn but not compared.
    ref = tmp_path / "ref"
    ref.mkdir()
    # Like the real export, the file holds (value, p) pairs, each value v
    # digitized on 0..1 and standing for 0.564 + 0.436 v (VALUE_AXIS): v0
    # stands for our accuracy 0.5 at p = 0, and v = 1 for 1 at p = 1.
    v0 = (0.5 - 0.564) / 0.436
    (ref / "barratt2022_fig2b_accuracy.csv").write_text(
        f"L = 6,,L = 20,\nX,Y,X,Y\n{v0},0.0,{v0},0.0\n1.0,1.0,0.9,0.5\n"
    )
    comparison = _load(PLOT).plot_sweep(sweep, ref)
    stats = comparison["accuracy"]["N6"]["N = 6"]
    assert stats["n_points"] == 2
    assert stats["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert "N = 20" not in comparison["accuracy"]["N6"]
    assert "N4-N6" in comparison["binder_crossings"]["P_corr_binder"]

    # Fig. 2c (issue #115): the mean entropy is ln 2 at p = 0 and 0 at
    # p = 1, which the fake reference says too, exported as (value, p) like
    # the real one; the Fig. 2b comparison file is left alone.
    (ref / "barratt2022_fig2c_entropy.csv").write_text(
        "L = 6,,L = 20,\nX,Y,X,Y\n0.0,1.0,0.0,1.0\n0.6931471805599453,0.0,0.1,0.5\n"
    )
    plot = _load(PLOT)
    panels, transition, filename, comparison_name = plot.FIGURES["2c"]
    assert transition == (r"$p_{success}$", 0.2)
    comparison = plot.plot_sweep(
        sweep,
        ref,
        panels,
        filename,
        transition=transition,
        comparison_name=comparison_name,
    )
    stats = comparison["entropy_mean"]["N6"]["N = 6"]
    assert stats["n_points"] == 2
    assert stats["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert "N4-N6" in comparison["binder_crossings"]["entropy_binder"]
    saved = json.loads((sweep / "comparison_fig2c.json").read_text())
    assert "entropy_mean" in saved
    assert "accuracy" in json.loads((sweep / "comparison.json").read_text())


def test_import_keeps_the_notebook_backend() -> None:
    """Importing the plot script leaves the backend alone (issue #144).

    It used to call ``matplotlib.use("Agg")`` at import: the notebooks that
    import it lost their inline backend, and their ``plt.show()`` warned
    "FigureCanvasAgg is non-interactive" instead of drawing the figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    previous = matplotlib.get_backend()
    plt.switch_backend("svg")  # stands for the notebook's inline backend
    try:
        _load(PLOT)
        assert matplotlib.get_backend() == "svg"
    finally:
        plt.switch_backend(previous)
