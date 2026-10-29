"""The charge-sharpening benchmark in experiments/charge_sharpening/ (issue #137)."""

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
from QELM.encoding import AncillaChargeSectorEncoding, product_charge_index
from QELM.evaluation import von_neumann_entropy

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "charge_sharpening"
SCRIPT = FOLDER / "charge_sharpening.py"
PLOT = FOLDER / "plot_charge_sharpening.py"
BASH_SCRIPT = FOLDER / "charge_sharpening.sh"
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
        ["samples.L=40", "charge_sharpening.n_boot=20", *overrides],
        config_name="charge_sharpening",
    )


def test_product_charge_index() -> None:
    """|010101> (Q = 3) et |010100> (Q = 2) pour 6 qubits, comme dans l'article."""
    assert product_charge_index(6, 3) == 0b010101
    assert product_charge_index(6, 2) == 0b010100
    assert product_charge_index(5, 2) == 0b01010
    assert product_charge_index(5, 1) == 0b01000
    assert product_charge_index(4, 0) == 0
    with pytest.raises(ValueError):
        product_charge_index(6, 4)


def test_product_encoding_is_eq_b1() -> None:
    """(|0>|010101> + |1>|010100>) / sqrt(2), avec R maximalement intriqué."""
    emb = AncillaChargeSectorEncoding({"N": 6, "initial": "product"})
    psi = emb.state()
    expected = np.zeros(2**7, dtype=complex)
    expected[0b0010101] = expected[0b1010100] = 1 / np.sqrt(2)
    np.testing.assert_allclose(psi, expected)
    assert von_neumann_entropy(psi, [0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        AncillaChargeSectorEncoding({"N": 6, "initial": "random"})
    # The default stays the uniform superposition.
    assert AncillaChargeSectorEncoding({"N": 6}).initial == "uniform"


def test_config_composes_with_the_paper_parameters() -> None:
    cfg = load_config(config_name="charge_sharpening")
    # The paper's L = 6 system qubits, plus the ancilla (issue #235).
    assert (cfg.N, cfg.Nr, cfg.N_total) == (6, 1, 7)
    assert cfg.samples.L == 10000
    assert cfg.U1_circuit.depth == 3
    assert cfg.ancilla_charge_sector.initial == "product"
    assert "projective_measurement" not in cfg
    assert cfg.weak_measurement.Nr == 1


def test_run_shapes_and_ranges() -> None:
    out = _load(SCRIPT).run(_cfg("weak_measurement.gamma=0.4"))
    assert out["S_R"].shape == (40,)
    assert out["M"].shape == (40, 2, 6)
    assert np.all((out["S_R"] >= -1e-12) & (out["S_R"] <= 1 + 1e-12))
    assert set(np.unique(out["M"])) <= {0.0, 1.0}
    assert out["gamma"] == 0.4
    for key in _load(SCRIPT).METRICS:
        assert np.isfinite(out[key]) or key.startswith("binder")


def test_no_measurement_keeps_the_ancilla_maximally_entangled() -> None:
    """gamma = 0 : rien n'est appris, S_R = ln 2 dans chaque trajectoire."""
    out = _load(SCRIPT).run(_cfg("weak_measurement.gamma=0.0"))
    np.testing.assert_allclose(out["S_R"], np.log(2), atol=1e-10)
    assert out["S_R_var"] == pytest.approx(0.0, abs=1e-12)
    assert out["click_rate"] == 0.0


def test_projective_measurement_reveals_the_charge() -> None:
    """gamma = 1 : la charge de S est mesurée, R est purifié, S_R = 0."""
    out = _load(SCRIPT).run(_cfg("weak_measurement.gamma=1.0"))
    np.testing.assert_allclose(out["S_R"], 0.0, atol=1e-10)
    # Each projective record of the 6 qubits holds charge 3 or 2 after the
    # first layer: the measured charge is the ancilla's branch.
    assert set(out["M"][:, 0].sum(axis=1)) <= {2.0, 3.0}


def test_entropy_decreases_with_gamma() -> None:
    """S_R moyen décroît quand la mesure se renforce (charge sharpening)."""
    run = _load(SCRIPT).run
    means = [
        run(_cfg(f"weak_measurement.gamma={g}", "samples.L=300"))["S_R_mean"]
        for g in (0.2, 0.5, 0.8)
    ]
    assert means[0] > means[1] > means[2]


def test_entropy_in_bits() -> None:
    out = _load(SCRIPT).run(
        _cfg("weak_measurement.gamma=0.0", "charge_sharpening.entropy_base=2")
    )
    np.testing.assert_allclose(out["S_R"], 1.0, atol=1e-9)


def test_measurement_rounds() -> None:
    """t - 1 rondes entre les couches, t avec measure_after_last_layer."""
    run = _load(SCRIPT).run
    assert run(_cfg("weak_measurement.gamma=0.5"))["M"].shape == (40, 2, 6)
    out = run(
        _cfg(
            "weak_measurement.gamma=0.5",
            "charge_sharpening.measure_after_last_layer=true",
        )
    )
    assert out["M"].shape == (40, 3, 6)


def test_same_seed_same_run_and_fixed_instance_shared_across_gamma() -> None:
    run = _load(SCRIPT).run
    a = run(_cfg("weak_measurement.gamma=0.5", "seed=3"))
    b = run(_cfg("weak_measurement.gamma=0.5", "seed=3"))
    np.testing.assert_array_equal(a["S_R"], b["S_R"])
    c = run(_cfg("weak_measurement.gamma=0.5", "seed=4"))
    assert not np.allclose(a["S_R"], c["S_R"])
    d = run(_cfg("weak_measurement.gamma=0.5", "U1_circuit.fixed_instance=false"))
    assert d["S_R"].shape == (40,)


def test_bash_script_and_plot(tmp_path: Path) -> None:
    """Le lanceur fait 1 N x 1 seed x 2 gamma, puis la figure et la comparaison."""
    assert os.access(BASH_SCRIPT, os.X_OK) and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "20",
            "GAMMA_VALUES": "0.0 1.0",
            "VIRTUAL_ENV": str(FOREIGN_VENV),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 1 x 1 x 2 runs" in result.stdout
    sweep = next((tmp_path / "charge_sharpening").iterdir())
    runs = sorted(p.name for p in sweep.iterdir() if p.is_dir())
    assert runs == ["N6_seed0_g0.0", "N6_seed0_g1.0"]
    metrics = json.loads((sweep / "N6_seed0_g1.0" / "metrics.json").read_text())
    assert metrics["gamma"] == 1.0 and metrics["S_R_mean"] == pytest.approx(
        0.0, abs=1e-9
    )
    # depth: ${int:${div:${N},2}} in the YAML, a number in the run (#85).
    resolved = json.loads(
        (sweep / "N6_seed0_g1.0" / "resolved_config.json").read_text()
    )
    assert resolved["U1_circuit"]["depth"] == 3
    assert (sweep / "charge_sharpening.png").is_file()

    # Against a fake reference S_R = ln2 (1 - gamma), which our two points
    # match, written as (value, gamma) like the digitized exports.
    ref = tmp_path / "ref"
    ref.mkdir()
    ln2 = np.log(2)
    (ref / "agrawal2024_fig7_S_R.csv").write_text(
        f"L=6,\nX,Y\n{ln2},0.0\n{ln2 / 2},0.5\n0.0,1.0\n"
    )
    comparison = _load(PLOT).plot_sweep(sweep, ref)
    # The paper's "L=6" (system sites) is our N = 6 (issue #235).
    stats = comparison["S_R_mean"]["N6"]["N = 6"]
    assert stats["n_points"] == 3
    assert stats["max_abs_diff"] == pytest.approx(0.0, abs=1e-9)
