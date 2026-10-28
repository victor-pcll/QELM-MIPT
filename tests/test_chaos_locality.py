"""Chaos and locality diagnostics of the reservoirs (issue #243)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest
import scipy.sparse as sp

from QELM.evaluation.chaos_locality import (
    R_GOE,
    R_POISSON,
    bell_reference_state,
    brickwork_layers,
    central_window,
    charge_operator,
    entanglement_growth,
    floquet_ising_gate,
    gap_ratios,
    goe_pdf,
    gue_pdf,
    ising_chain_hamiltonian,
    level_statistics,
    neel_state,
    otoc_map,
    parity_operator,
    poisson_pdf,
    propagator_operator,
    quasienergies,
    reference_entropy,
    reference_qubit_cutoff,
    reference_qubit_trajectories,
    reflection_operator,
    sector_spectra,
    step_unitary,
    symmetry_blocks,
    unfold,
    xxz_chain_hamiltonian,
    y_polarized_state,
)
from QELM.evaluation.entropy import von_neumann_entropy
from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.physics.operators import get_site_operator, pauli

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
X, Y, Z = pauli("X"), pauli("Y"), pauli("Z")


def _site(op: np.ndarray, i: int, N: int) -> np.ndarray:
    return get_site_operator(op, i, N)


def _commutator_norm(A: sp.spmatrix | np.ndarray, B: sp.spmatrix | np.ndarray) -> float:
    A = A.toarray() if sp.issparse(A) else A
    B = B.toarray() if sp.issparse(B) else B
    return float(np.abs(A @ B - B @ A).max())


# --- reference distributions -------------------------------------------------


@pytest.mark.parametrize("pdf", [poisson_pdf, goe_pdf, gue_pdf])
def test_reference_distributions_are_normalized_with_mean_one(pdf) -> None:
    s = np.linspace(0, 30, 300_001)
    P = pdf(s)
    assert np.trapezoid(P, s) == pytest.approx(1, abs=1e-6)
    assert np.trapezoid(s * P, s) == pytest.approx(1, abs=1e-6)


# --- Hamiltonians and symmetries ---------------------------------------------


def test_ising_chain_matches_the_kronecker_products() -> None:
    J = np.array([0.7, 1.3, 0.9])
    N = 4
    expected = sum(
        -J[i] * _site(Z, i, N) @ _site(Z, i + 1, N) for i in range(N - 1)
    ) + sum(-1.05 * _site(X, i, N) - 0.5 * _site(Z, i, N) for i in range(N))
    H = ising_chain_hamiltonian(J, h_x=1.05, h_z=0.5)
    np.testing.assert_allclose(H.toarray(), expected, atol=1e-12)


def test_xxz_chain_matches_the_kronecker_products_and_conserves_q() -> None:
    N, J, Delta = 4, 0.8, 1.3
    fields = np.array([0.1, -0.2, 0.3, 0.05])
    expected = sum(
        J * (_site(X, i, N) @ _site(X, i + 1, N) + _site(Y, i, N) @ _site(Y, i + 1, N))
        + Delta * _site(Z, i, N) @ _site(Z, i + 1, N)
        for i in range(N - 1)
    ) + sum(fields[i] * _site(Z, i, N) for i in range(N))
    H = xxz_chain_hamiltonian(N, J, Delta, fields)
    np.testing.assert_allclose(H.toarray(), expected, atol=1e-12)
    assert _commutator_norm(H, charge_operator(N)) < 1e-12
    periodic = xxz_chain_hamiltonian(N, J, Delta, boundary="periodic")
    extra = periodic - xxz_chain_hamiltonian(N, J, Delta)
    expected_bond = J * (
        _site(X, 3, N) @ _site(X, 0, N) + _site(Y, 3, N) @ _site(Y, 0, N)
    ) + Delta * _site(Z, 3, N) @ _site(Z, 0, N)
    np.testing.assert_allclose(extra.toarray(), expected_bond, atol=1e-12)


def test_symmetry_operators() -> None:
    N = 4
    PZ = parity_operator(N, "Z").toarray()
    PX = parity_operator(N, "X").toarray()
    R = reflection_operator(N).toarray()
    expected_PX = _site(X, 0, N) @ _site(X, 1, N) @ _site(X, 2, N) @ _site(X, 3, N)
    np.testing.assert_allclose(PX, expected_PX)
    np.testing.assert_allclose(PZ, np.diag(np.diag(PZ)))
    for S in (PZ, PX, R):
        np.testing.assert_allclose(S @ S, np.eye(2**N))
        np.testing.assert_allclose(S, S.conj().T)
    # R maps Z_0 to Z_{N-1}.
    np.testing.assert_allclose(R @ _site(Z, 0, N) @ R, _site(Z, N - 1, N))
    with pytest.raises(ValueError):
        parity_operator(N, "Y")


def test_the_symmetries_commute_with_their_models() -> None:
    N = 5
    J = np.array([0.6, 1.4, 1.1, 0.9])
    PX = parity_operator(N, "X")
    assert _commutator_norm(ising_chain_hamiltonian(J, 1.05, 0.0), PX) < 1e-12
    assert _commutator_norm(ising_chain_hamiltonian(J, 1.05, 0.5), PX) > 0.1
    np.random.seed(0)
    tfim = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": 10.0}).hamiltonian()
    assert _commutator_norm(tfim, parity_operator(N, "Z")) < 1e-12
    xxz = xxz_chain_hamiltonian(N)
    for S in (charge_operator(N), reflection_operator(N), PX):
        assert _commutator_norm(xxz, S) < 1e-12
    period = np.linalg.multi_dot(brickwork_layers(6, floquet_ising_gate())[::-1])
    assert _commutator_norm(period, reflection_operator(6)) < 1e-12


def test_symmetry_blocks_of_the_xxz_chain() -> None:
    N = 4
    blocks = symmetry_blocks(
        2**N, [charge_operator(N), reflection_operator(N), parity_operator(N, "X")]
    )
    assert sum(V.shape[1] for V in blocks) == 2**N
    for V in blocks:
        np.testing.assert_allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=1e-10)
    # Q = 1 (1000, 0100, 0010, 0001) is split by the reflection only, 2 + 2;
    # Q = 2 also by the spin flip, which maps Q to N - Q elsewhere and is
    # skipped there: (R, F) = (+, +) 3, (+, -) 1, (-, -) 2.
    Q = np.diag(charge_operator(N).toarray()).real
    by_charge: dict[int, list[int]] = {}
    for V in blocks:
        q = int(round(float(np.real(np.sum(np.abs(V) ** 2, axis=1) @ Q) / V.shape[1])))
        by_charge.setdefault(q, []).append(V.shape[1])
    assert sorted(by_charge[2]) == [1, 2, 3]
    assert sorted(by_charge[1]) == [2, 2]


@pytest.mark.parametrize(
    "make", [reflection_operator, lambda N: parity_operator(N, "X")]
)
def test_permutation_symmetries_split_into_their_eigenspaces(make) -> None:
    """The +1 / -1 bases built without eigh (spin flip, reflection)."""
    N = 5
    S = make(N).toarray()
    blocks = symmetry_blocks(2**N, [make(N)])
    assert sum(V.shape[1] for V in blocks) == 2**N
    for V, sign in zip(blocks, (1, -1), strict=True):
        np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-12)
        np.testing.assert_allclose(S @ V, sign * V, atol=1e-12)
    H = ising_chain_hamiltonian(np.linspace(0.6, 1.4, N - 1), 1.05, 0.0)
    spectra = sector_spectra(H, [parity_operator(N, "X")])
    np.testing.assert_allclose(
        np.sort(np.concatenate(spectra)), np.linalg.eigvalsh(H.toarray()), atol=1e-10
    )


def test_sector_spectra_recover_the_full_spectrum() -> None:
    N = 6
    H = xxz_chain_hamiltonian(N, fields=np.linspace(-0.3, 0.4, N))
    spectra = sector_spectra(H, [charge_operator(N)])
    assert [len(s) for s in spectra] == [math.comb(N, q) for q in range(N + 1)]
    np.testing.assert_allclose(
        np.sort(np.concatenate(spectra)), np.linalg.eigvalsh(H.toarray()), atol=1e-10
    )


# --- level statistics ---------------------------------------------------------


def test_gap_ratio_and_unfolding_on_known_spectra() -> None:
    rng = np.random.default_rng(1)
    np.testing.assert_allclose(gap_ratios(np.arange(10.0)), 1.0)
    poisson = np.cumsum(rng.exponential(size=50_000))
    assert gap_ratios(poisson).mean() == pytest.approx(R_POISSON, abs=0.01)
    goe_r, goe_s = [], []
    for _ in range(20):
        A = rng.normal(size=(300, 300))
        levels = central_window(np.linalg.eigvalsh(A + A.T))
        goe_r.append(gap_ratios(levels))
        goe_s.append(unfold(levels))
    assert np.concatenate(goe_r).mean() == pytest.approx(R_GOE, abs=0.015)
    s = np.concatenate(goe_s)
    assert s.mean() == pytest.approx(1, abs=1e-9)
    # Level repulsion: almost no small spacing, unlike exp(-s).
    assert np.mean(s < 0.1) < 0.02


def test_central_window_keeps_the_middle_third() -> None:
    np.testing.assert_allclose(central_window(np.arange(9.0)), [3, 4, 5])


def test_the_tilted_field_breaks_integrability() -> None:
    """Issue #243 A: Poisson at h_z = 0, Wigner-Dyson at h_z = 0.5."""
    rng = np.random.default_rng(2)
    N = 9
    stats = {}
    for h_z in (0.0, 0.5):
        spectra = []
        for _ in range(4):
            H = ising_chain_hamiltonian(1 + rng.random(N - 1) - 0.5, 1.05, h_z)
            symmetries = [parity_operator(N, "X")] if h_z == 0 else []
            spectra.append(sector_spectra(H, symmetries))
        stats[h_z] = level_statistics(spectra).mean_r
    assert stats[0.0] < 0.44
    assert stats[0.5] > 0.50


def test_quasienergies_of_a_floquet_period() -> None:
    N = 6
    layers = brickwork_layers(N, floquet_ising_gate())
    spectra = quasienergies(layers[1] @ layers[0], [reflection_operator(N)])
    assert sum(len(s) for s in spectra) == 2**N
    for theta in spectra:
        assert np.all(np.diff(theta) >= 0)
        assert np.all((theta >= -np.pi - 1e-12) & (theta <= np.pi + 1e-12))
    stats = level_statistics([spectra], circular=True)
    assert stats.spacings.mean() == pytest.approx(1)


# --- locality: entanglement growth and OTOC ----------------------------------


def test_entanglement_growth_from_the_neel_state() -> None:
    N = 6
    psi = neel_state(N)
    assert np.flatnonzero(psi).tolist() == [int("010101", 2)]
    S = entanglement_growth([np.eye(2**N)], psi, 3)
    np.testing.assert_allclose(S, 0, atol=1e-12)
    H = ising_chain_hamiltonian(np.ones(N - 1), 1.05, 0.5)
    S = entanglement_growth([step_unitary(H, 0.3)], psi, 20)
    assert S[0] == pytest.approx(0, abs=1e-12)
    assert np.all(S <= N / 2 + 1e-9) and S.max() > 1


def test_propagator_operator_matches_the_dense_unitary() -> None:
    N = 5
    H = ising_chain_hamiltonian(np.ones(N - 1), 1.05, 0.5)
    psi = neel_state(N)
    np.testing.assert_allclose(
        propagator_operator(H, 0.3) @ psi, step_unitary(H, 0.3) @ psi, atol=1e-10
    )
    dense = entanglement_growth([step_unitary(H, 0.3)], psi, 5)
    sparse = entanglement_growth([propagator_operator(H, 0.3)], psi, 5)
    np.testing.assert_allclose(sparse, dense, atol=1e-9)


def test_y_polarized_state_sits_at_zero_energy() -> None:
    N = 5
    psi = y_polarized_state(N)
    assert np.vdot(psi, psi).real == pytest.approx(1)
    assert von_neumann_entropy(psi, [0, 1]) == pytest.approx(0, abs=1e-12)
    np.random.seed(1)
    for H in (
        ising_chain_hamiltonian(np.linspace(0.6, 1.4, N - 1), 1.05, 0.5),
        TransverseFieldIsingModel({"N": N, "J": 1.0, "h": 10.0}).hamiltonian(),
    ):
        assert np.vdot(psi, H @ psi).real == pytest.approx(0, abs=1e-12)


def test_otoc_has_an_exact_light_cone_in_the_brickwork() -> None:
    N, x0 = 8, 4
    layers = brickwork_layers(N, floquet_ising_gate())
    C = otoc_map(layers, x0, 4)
    np.testing.assert_allclose(C[0], 0, atol=1e-12)
    for t in range(5):
        for x in range(N):
            # One half-layer moves an operator by at most one site.
            if abs(x - x0) > t + 1:
                assert C[t, x] < 1e-12, (t, x)
    assert C[4, x0 + 2] > 1e-3
    assert np.all(C <= 2 + 1e-9)


def test_otoc_of_the_all_to_all_tfim_reaches_every_site_at_once() -> None:
    np.random.seed(3)
    N = 6
    H = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": 1.0}).hamiltonian()
    C = otoc_map([step_unitary(H, 0.5)], 0, 1)
    assert np.all(C[1, 1:] > 1e-4)


def test_otoc_vanishes_for_commuting_dynamics() -> None:
    N = 4
    H = sum(_site(Z, i, N) @ _site(Z, i + 1, N) for i in range(N - 1))
    C = otoc_map([step_unitary(H, 0.7)], 1, 3)
    np.testing.assert_allclose(C, 0, atol=1e-12)


# --- reference qubit ------------------------------------------------------------


def test_bell_reference_state() -> None:
    M = bell_reference_state(4, 2)
    assert np.sum(np.abs(M) ** 2) == pytest.approx(1)
    assert reference_entropy(M) == pytest.approx(1)
    # Neel 0101 with site 2 (value 0) correlated with R.
    assert M[int("0101", 2), 0] == pytest.approx(1 / math.sqrt(2))
    assert M[int("0111", 2), 1] == pytest.approx(1 / math.sqrt(2))


def test_reference_qubit_limits() -> None:
    rng = np.random.default_rng(4)
    N, x0 = 6, 3
    layers = brickwork_layers(N, floquet_ising_gate())
    unitary = reference_qubit_trajectories(layers, x0, 0.0, 4, 2, rng)
    np.testing.assert_allclose(unitary.S_Q, 1, atol=1e-10)
    # At t = 0 only the measurement of x0 purifies R.
    expected = np.zeros(N)
    expected[x0] = -1
    np.testing.assert_allclose(unitary.delta_S_Q[0], expected, atol=1e-10)
    # Strict light cone of the brickwork: no change outside |x - x0| <= t + 1.
    for t in range(5):
        for x in range(N):
            if abs(x - x0) > t + 1:
                assert abs(unitary.delta_S_Q[t, x]) < 1e-10
    assert np.all(unitary.delta_S_Q <= 1e-10)
    # Every qubit measured: the system is a product state, R is pure.
    full = reference_qubit_trajectories(layers, x0, 1.0, 3, 2, rng)
    np.testing.assert_allclose(full.S_Q[1:], 0, atol=1e-10)


def test_reference_qubit_cutoff_limits() -> None:
    rng = np.random.default_rng(5)
    N, x0 = 4, 2
    H = ising_chain_hamiltonian(np.ones(N - 1), 1.05, 0.5)
    steps = [step_unitary(H, 0.3)]
    S = reference_qubit_cutoff(steps, x0, 1.0, 3, [-1, N], 2, rng)
    # Nothing recorded: rho_R stays maximally mixed.
    np.testing.assert_allclose(S[0], 1, atol=1e-10)
    # Everything recorded at p = 1: R is purified at the first step.
    np.testing.assert_allclose(S[1, 1:], 0, atol=1e-10)
    assert S[1, 0] == pytest.approx(1)


def test_reference_qubit_cutoff_matches_the_pure_trajectories() -> None:
    """With every outcome kept the density matrix stays pure: same mean."""
    N, x0, p, n = 4, 2, 0.3, 4
    H = ising_chain_hamiltonian(np.ones(N - 1), 1.05, 0.5)
    steps = [step_unitary(H, 0.3)]
    S_cut = reference_qubit_cutoff(steps, x0, p, n, [N], 300, np.random.default_rng(6))
    pure = reference_qubit_trajectories(steps, x0, p, n, 300, np.random.default_rng(7))
    np.testing.assert_allclose(S_cut[0], pure.S_Q, atol=0.08)


def test_reference_qubit_is_reproducible() -> None:
    layers = brickwork_layers(4, floquet_ising_gate())
    a = reference_qubit_trajectories(layers, 2, 0.3, 3, 3, np.random.default_rng(8))
    b = reference_qubit_trajectories(layers, 2, 0.3, 3, 3, np.random.default_rng(8))
    np.testing.assert_array_equal(a.S_Q, b.S_Q)
    np.testing.assert_array_equal(a.delta_S_Q, b.delta_S_Q)


# --- script --------------------------------------------------------------------


def _load_script() -> ModuleType:
    path = REPO_ROOT / "experiments" / "diagnostics" / "diagnostics_physiques.py"
    spec = importlib.util.spec_from_file_location("diagnostics_physiques", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnostics_physiques"] = module
    spec.loader.exec_module(module)
    return module


def test_script_writes_every_figure(tmp_path: Path) -> None:
    script = _load_script()
    out = script.main(["--quick", "--out", str(tmp_path)])
    for name in (
        "level_spacing",
        "entropy_growth",
        "otoc",
        "reference_qubit_delta",
        "reference_qubit_cutoff",
    ):
        assert (out / f"{name}.png").is_file()
        data = json.loads((out / f"{name}.json").read_text())
        assert len(data["panels"]) >= 5
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["spectra"]) == {
        "tfim_all_to_all",
        "ising_hz0",
        "ising_hz05",
        "xxz",
        "floquet",
    }
    assert set(summary["growth"]["ising_hz05"]) == {"neel", "y"}


def test_script_runs_one_part_alone(tmp_path: Path) -> None:
    script = _load_script()
    out = script.main(
        ["--quick", "--out", str(tmp_path), "--parts", "otoc", "--models", "floquet"]
    )
    assert sorted(p.name for p in out.glob("*.png")) == ["otoc.png"]
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"args", "otoc"}
    assert set(summary["otoc"]) == {"floquet"}
