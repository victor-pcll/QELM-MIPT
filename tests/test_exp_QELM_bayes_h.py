"""The learnability zone of the unitary QELM along h (issue #194)."""

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
from QELM.encoding import AncillaChargeSectorEncoding

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "QELM_bayes_h"
SCRIPT = FOLDER / "exp_QELM_bayes_h.py"
PLOT = FOLDER / "plot_bayes_h.py"
BASH_SCRIPT = FOLDER / "exp_QELM_bayes_h.sh"
QELM_Q = REPO_ROOT / "experiments" / "QELM_Q" / "exp_QELM_Q.py"
FOREIGN_VENV = REPO_ROOT.parent / "other-project" / ".venv"
L, N = 200, 4


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*extra: str) -> dict:
    cfg = load_config([f"N={N}", f"samples.L={L}", *extra], config_name="QELM_bayes_h")
    return _load(SCRIPT).run(cfg)


def test_config_composes() -> None:
    cfg = load_config(config_name="QELM_bayes_h")
    assert (cfg.N, cfg.Ns, cfg.Nr) == (4, 3, 1)
    assert cfg.models == "TFIM" and cfg.TFIM.h == 10.0
    assert cfg.evolution.t == 10.0  # t = 10 / J
    assert cfg.samples.L == 2000 and cfg.samples.train_size == 1400
    assert cfg.ancilla_charge_sector.initial == "uniform"
    assert cfg.bayes.reg_param == 1e-6 and cfg.bayes.entropy_base == 2


def test_sector_states_are_the_two_branches() -> None:
    sector = AncillaChargeSectorEncoding({"N": 5})
    psi1, psi2 = _load(SCRIPT).sector_states(sector)
    assert psi1.shape == psi2.shape == (16,)
    for psi, Q in ((psi1, sector.Q1), (psi2, sector.Q2)):
        assert np.linalg.norm(psi) == pytest.approx(1.0)
        support = np.flatnonzero(np.abs(psi) > 0)
        assert all(bin(b).count("1") == Q for b in support)


def test_shapes_and_ranges() -> None:
    out = _run("TFIM.h=0.5")
    assert out["X"].shape == (L, N - 1 + 1)
    np.testing.assert_array_equal(out["X"][:, -1], 1.0)
    assert np.all(np.abs(out["X"][:, :-1]) <= 1 + 1e-12)
    assert set(np.unique(out["Y"])) == {0.0, 1.0}  # Ns = 3: Q1 = 1, Q2 = 0
    n_test = L - int(0.7 * L)
    assert out["y_pred"].shape == out["P_corr_bayes"].shape == (n_test,)
    for key in _load(SCRIPT).EXTRA_METRICS:
        assert 0.0 <= out[key] <= 1.0, key
    for P in (out["P_corr_ridge"], out["P_corr_bayes"]):
        assert np.all((P >= 0) & (P <= 1))


def test_same_samples_as_the_qelm_q_at_p_zero() -> None:
    """Same disorder, labels and <Z_k> as exp_QELM_Q without measurement,
    which evolves the full density matrix of R + S with a dense U."""
    ours = _run("TFIM.h=2")
    cfg = load_config(
        [
            "models=TFIM",
            f"N={N}",
            f"samples.L={L}",
            "TFIM.h=2",
            "projective_measurement.p=0.0",
            "projective_measurement.nbr_time_evolution=1",
        ]
    )
    theirs = _load(QELM_Q).run(cfg)
    np.testing.assert_array_equal(ours["Y"], theirs["Y"])
    np.testing.assert_allclose(ours["X"], theirs["X"], atol=1e-8)
    np.testing.assert_allclose(ours["y_pred"], theirs["y_pred"], atol=1e-6)
    assert ours["accuracy_ridge"] == theirs["accuracy"]
    assert ours["accuracy_chance"] == theirs["accuracy_baseline"]


def test_large_field_conserves_the_charge() -> None:
    """At h = 10 J the X_i X_j couplings barely change Q, so sum_k <Z_k>
    stays at Ns - 2 Q and every classifier reads the charge."""
    out = _run("TFIM.h=10")
    Ns = N - 1
    np.testing.assert_allclose(
        out["X"][:, :-1].sum(axis=1), Ns - 2 * out["Y"], atol=0.05
    )
    assert out["accuracy_ridge"] == out["accuracy_bayes"] == 1.0
    assert out["entropy_bayes"] < 0.01


def test_bayes_beats_ridge_without_field() -> None:
    """At h = 0 the charge is no longer in sum_k <Z_k>: the Ridge falls to
    chance, while the QDA still reads it from the spread of the <Z_k>."""
    out = _run("TFIM.h=0", "samples.L=1000", "N=6")
    assert out["accuracy_ridge"] < 0.6
    assert out["accuracy_bayes"] > 0.7
    assert out["entropy_bayes"] < out["entropy_ridge"] <= out["entropy_chance"] + 0.01


def test_product_initial_state() -> None:
    out = _run("TFIM.h=1", "ancilla_charge_sector.initial=product")
    assert out["X"].shape == (L, N)


def test_reproducible_with_the_seed() -> None:
    a, b = _run("TFIM.h=1", "seed=3"), _run("TFIM.h=1", "seed=3")
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["P_corr_bayes"], b["P_corr_bayes"])
    c = _run("TFIM.h=1", "seed=4")
    assert not np.array_equal(a["X"], c["X"])


def test_bash_script_and_plot(tmp_path: Path) -> None:
    """The launcher runs 1 N x 2 h x 2 seeds, then one figure per N."""
    assert os.access(BASH_SCRIPT, os.X_OK) and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    out_dir = tmp_path / "sweep"
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_DIR": str(out_dir),
            "L": "60",
            "N_VALUES": "4",
            "H_VALUES": "0 10",
            "SEEDS": "0 1",
            "VIRTUAL_ENV": str(FOREIGN_VENV),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 1 x 2 x 2 runs" in result.stdout
    runs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert runs == ["N4_h0_seed0", "N4_h0_seed1", "N4_h10_seed0", "N4_h10_seed1"]
    metrics = json.loads((out_dir / "N4_h10_seed0" / "metrics.json").read_text())
    assert metrics["accuracy_bayes"] == 1.0
    assert (out_dir / "N4_h10_seed0" / "exp_QELM_bayes_h.npz").is_file()
    assert (out_dir / "learnability_zone_N4.png").is_file()
    data = json.loads((out_dir / "learnability_zone_N4.json").read_text())
    summary = data["extra"]["summary"]
    assert [row["h"] for row in summary] == [0.0, 10.0]
    assert all(row["n_runs"] == 2 for row in summary)
    assert summary[1]["accuracy_bayes_mean"] == 1.0


def test_import_keeps_the_notebook_backend() -> None:
    """Importing the plot script leaves the backend alone (issue #144)."""
    import matplotlib

    before = matplotlib.get_backend()
    _load(PLOT)
    assert matplotlib.get_backend() == before
