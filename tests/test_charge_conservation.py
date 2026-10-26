"""Charge conservation check and optional filtering (issue #134)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config
from QELM.encoding import AncillaChargeSectorEmbedding, charge_sector_state
from QELM.evaluation import charge_filter, conservation_metrics, conserved_mask
from QELM.evaluation.learnability import charge_state, sample_records
from QELM.models.circuits.base import apply_two_qubit_gate
from QELM.models.circuits.haar import U1BrickworkCircuit
from QELM.physics import (
    X,
    batch_charge_leakage,
    charge_distribution,
    charge_leakage,
    charge_values,
    get_site_operator,
    sharp_charge,
    total_charge,
)
from QELM.training import WeakMeasurement

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARPENING = REPO_ROOT / "experiments" / "charge_sharpening" / "charge_sharpening.py"
LEARNABILITY = REPO_ROOT / "experiments" / "learnability" / "learnability.py"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_state(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**N) + 1j * rng.standard_normal(2**N)
    return psi / np.linalg.norm(psi)


# --- Primitives -------------------------------------------------------------


def test_charge_values_counts_the_ones() -> None:
    """Q(x) est le nombre de bits à 1, le site 0 étant le bit de poids fort."""
    np.testing.assert_array_equal(charge_values(2), [0, 1, 1, 2])
    q = charge_values(3)
    assert q[0b101] == 2 and q[0b111] == 3 and q[0b000] == 0
    # Restricted to site 0 (the most significant bit) only.
    np.testing.assert_array_equal(charge_values(2, sites=[0]), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        charge_values(0)
    with pytest.raises(ValueError):
        charge_values(3, sites=[3])


def test_distribution_of_known_states() -> None:
    """P(Q) d'un état de base, d'un secteur et de |+>^N (binomiale)."""
    N = 4
    basis = np.zeros(2**N, dtype=complex)
    basis[0b0110] = 1
    np.testing.assert_allclose(charge_distribution(basis), [0, 0, 1, 0, 0])
    assert total_charge(basis) == 2
    sector = charge_sector_state(N, 3)
    np.testing.assert_allclose(charge_distribution(sector), [0, 0, 0, 1, 0])
    plus = np.full(2**N, 2 ** (-N / 2), dtype=complex)
    np.testing.assert_allclose(
        charge_distribution(plus), np.array([1, 4, 6, 4, 1]) / 16
    )
    assert total_charge(plus) == pytest.approx(N / 2)


def test_density_matrix_and_column_agree_with_the_vector() -> None:
    psi = _random_state(4)
    rho = np.outer(psi, psi.conj())
    p = charge_distribution(psi)
    np.testing.assert_allclose(charge_distribution(rho), p)
    np.testing.assert_allclose(charge_distribution(psi[:, None]), p)
    assert p.sum() == pytest.approx(1.0)
    assert charge_leakage(rho, 2) == pytest.approx(charge_leakage(psi, 2))
    # An unnormalized state gives the same probabilities.
    np.testing.assert_allclose(charge_distribution(3 * psi), p)
    with pytest.raises(ValueError):
        charge_distribution(np.zeros(8))


def test_sharp_charge() -> None:
    """La charge de Eq. (B1) sur R + S vaut Ns // 2 ; |+>^N n'en a pas."""
    for initial in ("uniform", "product"):
        psi = AncillaChargeSectorEmbedding({"N": 7, "initial": initial}).state()
        assert sharp_charge(psi) == 3
    with pytest.raises(ValueError, match="sharp"):
        sharp_charge(np.full(8, 8**-0.5))


def test_leakage_values() -> None:
    """0 dans le secteur, 1 hors du secteur, P(Q != Q0) entre les deux."""
    N = 4
    psi = charge_sector_state(N, 2)
    assert charge_leakage(psi, 2) == 0.0
    assert charge_leakage(psi, 1) == 1.0
    flipped = get_site_operator(X, 0, N) @ psi
    assert charge_leakage(flipped, 2) == 1.0
    mixed = (psi + charge_sector_state(N, 1)) / np.sqrt(2)
    assert charge_leakage(mixed, 2) == pytest.approx(0.5)


def test_u1_dynamics_conserves_the_charge_exactly() -> None:
    """Portes U(1), brouillage et mesure faible : fuite exactement nulle."""
    rng = np.random.default_rng(1)
    circuit = U1BrickworkCircuit(
        {"N": 7, "Nr": 1, "depth": 3, "scrambling_layers": 5}, rng
    )
    measurement = WeakMeasurement({"gamma": 0.5, "Nr": 1})
    psi = AncillaChargeSectorEmbedding({"N": 7, "initial": "product"}).state()
    Q = sharp_charge(psi)
    psi = circuit.scramble(psi)
    assert charge_leakage(psi, Q) == 0.0
    for t in range(circuit.depth):
        psi = circuit.apply_layer(psi, t)
        psi = measurement.measure(psi, rng=rng).rho
        assert charge_leakage(psi, Q) == 0.0
    # A gate that does not conserve the charge (|00> <-> |11>) is caught.
    swap_00_11 = np.eye(4, dtype=complex)[[3, 1, 2, 0]]
    assert charge_leakage(apply_two_qubit_gate(psi, swap_00_11, 1, 2), Q) > 0


def test_batch_leakage_matches_the_single_one() -> None:
    psi = np.stack([_random_state(4, s) for s in range(5)])
    Q = np.array([0, 1, 2, 3, 4])
    expected = [charge_leakage(p, int(q)) for p, q in zip(psi, Q, strict=True)]
    np.testing.assert_allclose(batch_charge_leakage(psi, Q), expected)
    with pytest.raises(ValueError):
        batch_charge_leakage(psi, Q[:3])


def test_sample_records_reports_zero_leakage() -> None:
    """Les enregistrements de Barratt et al. conservent la charge."""
    records = sample_records(6, 0.3, 32, rng=np.random.default_rng(0))
    assert records.charge_leakage is not None
    assert records.charge_leakage.shape == (32,)
    assert np.all(records.charge_leakage == 0.0)
    assert (
        batch_charge_leakage(
            np.stack([charge_state(6, int(q)) for q in records.charge]), records.charge
        ).max()
        == 0.0
    )


# --- Filtering --------------------------------------------------------------


def test_mask_keeps_everything_by_default() -> None:
    leak = np.array([0.0, 1e-3, 0.0, 1.0])
    np.testing.assert_array_equal(conserved_mask(leak, 1e-10, False), [1, 1, 1, 1])
    np.testing.assert_array_equal(conserved_mask(leak, 1e-10, True), [1, 0, 1, 0])
    np.testing.assert_array_equal(conserved_mask(leak, 1e-2, True), [1, 1, 1, 0])
    with pytest.raises(ValueError):
        conserved_mask(leak, -1.0, True)


def test_metrics_and_filter() -> None:
    leak = np.array([0.0, 0.5, 0.0, 0.0])
    kept = conserved_mask(leak, 1e-10, False)
    assert conservation_metrics(leak, 1e-10, kept) == {
        "max_charge_leakage": 0.5,
        "n_non_conserved": 1,
        "n_discarded": 0,
        "discarded_fraction": 0.0,
    }
    kept, metrics = charge_filter(
        leak, {"atol": 1e-10, "filter_non_conserved_charge": True}
    )
    np.testing.assert_array_equal(kept, [1, 0, 1, 1])
    assert metrics["n_discarded"] == 1 and metrics["discarded_fraction"] == 0.25
    # The flag defaults to False when absent.
    kept, _ = charge_filter(leak, {"atol": 1e-10})
    assert kept.all()
    with pytest.raises(ValueError, match="Every run"):
        charge_filter(np.ones(3), {"atol": 0.0, "filter_non_conserved_charge": True})


@pytest.mark.parametrize("config_name", ["charge_sharpening", "learnability"])
def test_configs_disable_the_filter_by_default(config_name: str) -> None:
    cfg = load_config(config_name=config_name)
    assert cfg.charge_conservation.filter_non_conserved_charge is False
    assert cfg.charge_conservation.atol == pytest.approx(1e-10)


# --- Experiments ------------------------------------------------------------


def _sharpening_cfg(*overrides: str):  # noqa: ANN202
    return load_config(
        ["samples.L=20", "charge_sharpening.n_boot=10", *overrides],
        config_name="charge_sharpening",
    )


def test_sharpening_run_checks_the_charge(tmp_path: Path) -> None:
    module = _load(SHARPENING)
    out = module.run(_sharpening_cfg("weak_measurement.gamma=0.4"))
    assert out["charge_leakage"].shape == (20,)
    assert np.all(out["charge_leakage"] == 0.0)
    assert out["kept"].all()
    assert out["n_non_conserved"] == 0 and out["n_discarded"] == 0
    module.save(out, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_discarded"] == 0 and metrics["max_charge_leakage"] == 0.0
    assert metrics["n_samples"] == 20
    with np.load(tmp_path / "charge_sharpening.npz") as data:
        assert data["kept"].all() and data["charge_leakage"].shape == (20,)


def _break_every_other_trajectory(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flip system qubit 1 after the first layer, one trajectory out of two."""
    original = U1BrickworkCircuit.apply_layer
    calls = {"n": 0}

    def apply_layer(self: U1BrickworkCircuit, state: np.ndarray, t: int) -> np.ndarray:
        state = original(self, state, t)
        if t == 0:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                state = get_site_operator(X, 1, self.N) @ state
        return state

    monkeypatch.setattr(module.U1BrickworkCircuit, "apply_layer", apply_layer)


@pytest.mark.parametrize("enabled", [False, True])
def test_sharpening_filter_drops_broken_trajectories(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Le filtre n'agit que si filter_non_conserved_charge vaut true."""
    module = _load(SHARPENING)
    _break_every_other_trajectory(module, monkeypatch)
    flag = f"charge_conservation.filter_non_conserved_charge={str(enabled).lower()}"
    out = module.run(_sharpening_cfg("weak_measurement.gamma=0.4", flag))
    broken = out["charge_leakage"] > 1e-10
    assert broken.sum() == 10
    assert out["n_non_conserved"] == 10
    if enabled:
        np.testing.assert_array_equal(out["kept"], ~broken)
        assert out["n_discarded"] == 10 and out["discarded_fraction"] == 0.5
        assert out["S_R_mean"] == pytest.approx(np.mean(out["S_R"][~broken]))
    else:
        assert out["kept"].all() and out["n_discarded"] == 0
        assert out["S_R_mean"] == pytest.approx(np.mean(out["S_R"]))


def _learnability_cfg(*overrides: str):  # noqa: ANN202
    return load_config(
        [
            "samples.L=64",
            "learnability.n_boot=10",
            "learnability.batch=32",
            "learnability.p=0.2",
            *overrides,
        ],
        config_name="learnability",
    )


def test_learnability_run_checks_the_charge(tmp_path: Path) -> None:
    module = _load(LEARNABILITY)
    out = module.run(_learnability_cfg())
    assert np.all(out["charge_leakage"] == 0.0) and out["kept"].all()
    assert out["n_discarded"] == 0
    module.save(out, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_records"] == 64 and metrics["n_non_conserved"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_learnability_filter_drops_broken_records(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    module = _load(LEARNABILITY)
    original = module.sample_records

    def broken_records(*args: object, **kwargs: object):  # noqa: ANN202
        records = original(*args, **kwargs)
        leak = np.zeros(records.charge.shape)
        leak[::4] = 1.0
        return dataclasses.replace(records, charge_leakage=leak)

    monkeypatch.setattr(module, "sample_records", broken_records)
    flag = f"charge_conservation.filter_non_conserved_charge={str(enabled).lower()}"
    out = module.run(_learnability_cfg(flag))
    assert out["n_non_conserved"] == 16
    if enabled:
        assert out["n_discarded"] == 16 and out["kept"].sum() == 48
        assert out["P_corr_mean"] == pytest.approx(np.mean(out["P_corr"][out["kept"]]))
    else:
        assert out["n_discarded"] == 0 and out["kept"].all()
