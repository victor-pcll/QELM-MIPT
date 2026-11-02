"""The continuous-time tilted-field Ising chain and its Poisson monitoring (#241)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM import load_config
from QELM.models.continuous_tfim_1d import ContinuousTFIMChain
from QELM.physics import X, Z, get_site_operator
from QELM.training.measurements.poisson import (
    micro_step_probability,
    poisson_monitored_batch,
    poisson_rate,
)


def _dense(N: int, J: np.ndarray, h_x: float, h_z: float) -> np.ndarray:
    """``H`` written with full 2**N operators, a different code path."""

    def site(P: np.ndarray, k: int) -> np.ndarray:
        return np.asarray(get_site_operator(P, k, N))

    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H -= J[i] * site(Z, i) @ site(Z, i + 1)
    for i in range(N):
        H -= h_x * site(X, i) + h_z * site(Z, i)
    return H


def test_hamiltonian_matches_its_definition() -> None:
    """Hermitian, and equal to - sum J_i Z_i Z_i+1 - h_x sum X - h_z sum Z on
    the nearest-neighbour bonds of the open chain only."""
    np.random.seed(3)
    model = ContinuousTFIMChain({"N": 5})
    H = model.hamiltonian().toarray()
    np.testing.assert_allclose(H, H.conj().T)
    np.testing.assert_allclose(H, _dense(5, model.couplings, 1.05, 0.5), atol=1e-12)
    assert model.bonds() == [(0, 1), (1, 2), (2, 3), (3, 4)]
    # Off the diagonal, only single bit flips (X_i), never two (no X X).
    rows, cols = np.nonzero(H - np.diag(np.diag(H)))
    assert set(np.unique([bin(r ^ c).count("1") for r, c in zip(rows, cols)])) == {1}


def test_couplings_are_uniform_in_half_to_three_halves() -> None:
    np.random.seed(0)
    J = np.concatenate([ContinuousTFIMChain({"N": 12}).couplings for _ in range(200)])
    assert J.min() >= 0.5 and J.max() <= 1.5
    assert J.mean() == pytest.approx(1.0, abs=0.02)
    np.random.seed(1)
    a = ContinuousTFIMChain({"N": 6}).couplings
    np.random.seed(1)
    np.testing.assert_array_equal(a, ContinuousTFIMChain({"N": 6}).couplings)
    clean = ContinuousTFIMChain({"N": 4, "W": 0.0}).couplings
    np.testing.assert_array_equal(clean, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="J_i > 0"):
        ContinuousTFIMChain({"N": 4, "W": 2.0})


def test_two_site_spectrum_and_longitudinal_term() -> None:
    """N = 2 with J = 1: without fields the spectrum is {-1, -1, 1, 1}; the
    longitudinal field shifts |00> by -2 h_z and |11> by +2 h_z (-2 and 0); with
    every term, the spectrum of the 4 x 4 matrix of the definition."""

    def spectrum(h_x: float, h_z: float) -> np.ndarray:
        model = ContinuousTFIMChain({"N": 2, "W": 0.0, "h_x": h_x, "h_z": h_z})
        return np.linalg.eigvalsh(model.hamiltonian().toarray())

    np.testing.assert_allclose(spectrum(0.0, 0.0), [-1, -1, 1, 1])
    np.testing.assert_allclose(spectrum(0.0, 0.5), [-2, 0, 1, 1])
    H = _dense(2, np.array([1.0]), 1.05, 0.5)
    np.testing.assert_allclose(spectrum(1.05, 0.5), np.linalg.eigvalsh(H), atol=1e-12)


def test_chaotic_level_statistics() -> None:
    """The mean ratio of consecutive level spacings in the middle of the
    spectrum is close to the GOE value 0.53, far from Poisson 0.39."""
    np.random.seed(0)
    E = np.linalg.eigvalsh(ContinuousTFIMChain({"N": 10}).hamiltonian().toarray())
    E = E[E.size // 4 : 3 * E.size // 4]
    s = np.diff(E)
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    assert r.mean() == pytest.approx(0.53, abs=0.03)


def test_rate_calibration() -> None:
    """gamma = -ln(1 - p) / tau, and n_sub micro-steps of probability
    1 - exp(-gamma dt) give p over a window."""
    p = np.array([0.0, 0.1, 0.5, 0.9])
    tau, n_sub = 0.3, 3
    gamma = poisson_rate(p, tau)
    np.testing.assert_allclose(gamma, -np.log(1 - p) / tau)
    q = micro_step_probability(p, n_sub)
    np.testing.assert_allclose(q, 1 - np.exp(-gamma * tau / n_sub))
    np.testing.assert_allclose(1 - (1 - q) ** n_sub, p)
    assert np.isinf(poisson_rate(1.0, tau)) and micro_step_probability(1.0, 4) == 1.0


def test_poisson_batch_limits() -> None:
    """p = 0: no measurement, the state is U^(n_windows n_sub) psi0. p = 1:
    every qubit measured in every window, a product state at the end. With
    U = 1 the share of measured (qubit, window) pairs is p."""
    np.random.seed(0)
    N, n_windows, n_sub = 4, 3, 2
    H = ContinuousTFIMChain({"N": N}).hamiltonian().toarray()
    E, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * E * 0.1)) @ V.conj().T
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0b0101] = 1.0
    rng = np.random.default_rng(0)
    M, psi = poisson_monitored_batch(
        U, psi0, np.array([0.0, 1.0]), n_windows, n_sub, rng
    )
    assert M.shape == (2, n_windows, N)
    np.testing.assert_array_equal(M[0], 0.0)
    expected = np.linalg.matrix_power(U, n_windows * n_sub) @ psi0
    np.testing.assert_allclose(psi[:, 0], expected, atol=1e-12)
    assert np.all(np.abs(M[1]) == 1.0)
    assert np.count_nonzero(np.abs(psi[:, 1]) > 1e-12) == 1
    p = np.full(400, 0.3)
    M, _ = poisson_monitored_batch(np.eye(2**N), psi0, p, 5, 3, rng)
    assert np.mean(M != 0) == pytest.approx(0.3, abs=0.02)
    with pytest.raises(ValueError, match="n_sub"):
        poisson_monitored_batch(U, psi0, p, 1, 0, rng)


def test_reads_its_config() -> None:
    cfg = load_config(["models=continuous_tfim_1d", "N=6"])
    assert cfg.models == "continuous_tfim_1d"
    block = cfg.continuous_tfim_1d
    assert (block.J, block.W, block.h_x, block.h_z) == (1.0, 1.0, 1.05, 0.5)
    model = ContinuousTFIMChain.from_config(cfg)
    assert model.N == 6 and model.couplings.shape == (5,)
