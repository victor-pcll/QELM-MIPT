"""Pauli strings as Bloch/Fourier monomials in the dense encoding (issue #364)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.encoding import DenseAngleEncoding
from QELM.evaluation.fourier_framework import (
    eigenstate_inputs,
    pauli_codes,
    pauli_matrix,
    pauli_vectors,
)
from QELM.evaluation.pauli_bloch import (
    brickwork_layers,
    dense_monomials,
    frequencies,
    gram_matrix,
    heisenberg_z,
    passage_matrix,
    pauli_label,
    pauli_labels,
    reconstruction_order,
    residual_variance,
    site_moments,
    sphere_weights,
    trig_index,
    trig_label,
    trig_monomials,
    weight_profile,
)
from QELM.models.circuits.base import apply_two_qubit_gate


def _inputs(n: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, np.pi, size=(k, 2 * n))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_monomials_are_pauli_expectations_of_the_dense_encoding(n: int) -> None:
    enc = DenseAngleEncoding({"N": n})
    x = _inputs(n, 12)
    B = dense_monomials(x)
    for xi, Bi in zip(x, B, strict=True):
        rho = enc.rho(xi)
        expect = np.array(
            [np.trace(pauli_matrix(c) @ rho).real for c in pauli_codes(n)]
        )
        assert np.allclose(Bi, expect, atol=1e-12)


def test_single_input_shape() -> None:
    x = _inputs(2, 1)[0]
    assert dense_monomials(x).shape == (16,)
    assert trig_monomials(x).shape == (81,)
    assert np.allclose(dense_monomials(x), dense_monomials(x[None])[0])


def test_odd_number_of_features_is_rejected() -> None:
    with pytest.raises(ValueError, match="two angles"):
        dense_monomials(np.zeros(3))


def test_any_observable_is_its_pauli_expansion() -> None:
    n = 3
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    obs = A + A.conj().T
    c = pauli_vectors(obs).real
    enc = DenseAngleEncoding({"N": n})
    x = _inputs(n, 10, seed=2)
    exact = np.array([np.trace(obs @ enc.rho(xi)).real for xi in x])
    assert np.allclose(dense_monomials(x) @ c, exact, atol=1e-10)


@pytest.mark.parametrize("n", [1, 2])
def test_passage_matrix_is_a_partial_permutation(n: int) -> None:
    M = passage_matrix(n)
    assert M.shape == (4**n, 9**n)
    assert np.all(M.sum(axis=1) == 1)
    assert np.all(M.sum(axis=0) <= 1)
    x = _inputs(n, 7, seed=3)
    assert np.allclose(dense_monomials(x), trig_monomials(x) @ M.T)


def test_labels() -> None:
    assert pauli_label((1, 0, 3)) == "XIZ"
    assert pauli_labels(1) == ["I", "X", "Y", "Z"]
    assert trig_label(trig_index((1,)), 1) == "sin θ0 cos φ0"
    assert trig_label(trig_index((0, 3)), 2) == "cos θ1"
    assert trig_label(0, 2) == "1"


def test_frequencies() -> None:
    k_theta, k_phi = frequencies(2)
    labels = pauli_labels(2)
    assert k_theta[labels.index("XZ")] == 2
    assert k_phi[labels.index("XZ")] == 1
    assert k_phi[labels.index("XY")] == 2
    assert k_theta[labels.index("II")] == 0


def test_weight_is_the_maximal_theta_frequency() -> None:
    # Along theta_i = theta, phi_i = 0, B_a = sin^{#X} cos^{#Z}: a polynomial
    # of degree exactly |a| in e^{i theta}, of the parity of |a|.
    n = 3
    s = 2 * np.pi * np.arange(64) / 64
    x = np.zeros((64, 2 * n))
    x[:, 0::2] = s[:, None]
    B = dense_monomials(x)
    k_theta, _ = frequencies(n)
    codes = pauli_codes(n)
    for a in range(4**n):
        if np.any(codes[a] == 2):
            assert np.allclose(B[:, a], 0.0)
            continue
        spec = np.abs(np.fft.rfft(B[:, a])) / 64
        support = np.flatnonzero(spec > 1e-10)
        assert support.max() == k_theta[a]
        assert np.all(support % 2 == k_theta[a] % 2)


def test_sphere_gram_matches_the_six_eigenstates() -> None:
    n = 2
    inputs = eigenstate_inputs(n, "sphere")
    B = inputs.monomials
    G = B.T @ B / len(B)
    assert np.allclose(G, gram_matrix(n, "sphere"))
    assert np.allclose(np.diag(gram_matrix(n, "sphere")), sphere_weights(n))


def test_box_moments_match_monte_carlo() -> None:
    G = site_moments("box")
    assert np.allclose(G, G.T)
    # Exact values: E[sin th] = 2/pi, E[sin phi] = 2/pi, E[cos phi] = 0.
    assert G[0, 2] == pytest.approx(4 / np.pi**2)
    assert G[0, 1] == pytest.approx(0.0, abs=1e-14)
    assert G[3, 3] == pytest.approx(0.5)
    x = np.random.default_rng(4).uniform(0, np.pi, size=(200_000, 2))
    B = dense_monomials(x)
    assert np.allclose(B.T @ B / len(B), G, atol=5e-3)
    with pytest.raises(ValueError, match="unknown"):
        site_moments("cube")


def test_heisenberg_feature_matches_the_forward_channel() -> None:
    n, p = 3, 0.3
    rng = np.random.default_rng(5)
    layers = brickwork_layers(n, 3, rng)
    c = heisenberg_z(layers, n, 1, p)
    enc = DenseAngleEncoding({"N": n})
    Z1 = pauli_matrix((0, 3, 0))
    for x in _inputs(n, 4, seed=6):
        rho = enc.rho(x)
        for layer in layers:
            for i, j, g in layer:
                rho = apply_two_qubit_gate(rho, g, i, j)
            # Forgotten Z measurement of each site with probability p.
            for site in range(n):
                Zs = pauli_matrix(tuple(3 if k == site else 0 for k in range(n)))
                rho = (1 - p / 2) * rho + (p / 2) * Zs @ rho @ Zs
        assert dense_monomials(x) @ c == pytest.approx(np.trace(Z1 @ rho).real)


def test_heisenberg_without_layers_is_z() -> None:
    c = heisenberg_z([], 2, 0, 0.5)
    assert c[pauli_labels(2).index("ZI")] == pytest.approx(1.0)
    assert np.abs(c).sum() == pytest.approx(1.0)


def test_weight_profile_and_parseval() -> None:
    n = 3
    c = heisenberg_z(brickwork_layers(n, 2, np.random.default_rng(7)), n, 0, 0.0)
    w = sphere_weights(n)
    P = weight_profile(c, n)
    # sum_a w_a c_a^2 = E[f^2] on the sphere (Parseval).
    G = gram_matrix(n, "sphere")
    assert P.sum() == pytest.approx(c @ G @ c)
    order = reconstruction_order(c, w)
    assert order[0] == 0
    assert sorted(order.tolist()) == list(range(4**n))
    power = w * c**2
    assert np.all(np.diff(power[order[1:]]) <= 1e-15)
    res = residual_variance(c, w, order)
    assert res[0] == pytest.approx(1.0)
    assert res[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(res) <= 1e-12)
    const = residual_variance(np.eye(4)[0], sphere_weights(1), np.arange(4))
    assert np.allclose(const, 0.0)


def test_figure_script_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import importlib.util
    from pathlib import Path

    folder = Path(__file__).parents[1] / "experiments" / "pauli_bloch"
    path = folder / "plot_pauli_bloch.py"
    spec = importlib.util.spec_from_file_location("plot_pauli_bloch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main([str(tmp_path), "--n", "3", "--circuits", "2", "--depth", "2"])
    for name in (
        "un_qubit",
        "deux_qubits",
        "matrice_passage",
        "verification",
        "reconstruction",
        "poids_frequence",
    ):
        assert (tmp_path / f"{name}.png").exists()
        assert (tmp_path / f"{name}.json").exists()
