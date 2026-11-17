"""Spectrum of the effects and ridge bias/variance (issue #334)."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import scipy.linalg

from QELM.evaluation.effect_mipt import draw_haar_circuit
from QELM.evaluation.fourier import bloch_monomials, transfer_matrix
from QELM.evaluation.fourier_tasks import product_state
from QELM.evaluation.ridge_spectrum import (
    approximation_bias,
    circle_weights,
    conditional_means,
    degree_target_circle,
    effect_span,
    generalized_spectrum,
    layers_from_gates,
    n_learn,
    record_spectrum,
    ridge_sweep,
    sample_effect,
    sample_shots,
    shot_groups,
    single_shot_r2,
    single_site_target,
    spectral_r2,
    target_values,
)

P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)


def _projector(n: int, site: int, outcome: int) -> np.ndarray:
    ops = [np.eye(2, dtype=complex)] * n
    ops[site] = P1 if outcome else P0
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _layer_unitary(gates: list[np.ndarray], n: int) -> np.ndarray:
    """Unitary of one layer of :func:`layers_from_gates`, brute force."""
    u = np.eye(2**n, dtype=complex)
    for i, _, gate in layers_from_gates([gates], n)[0]:
        full = np.kron(np.kron(np.eye(2**i), gate), np.eye(2 ** (n - i - 2)))
        u = full @ u
    return u


def _exact_effects(
    gates: list[list[np.ndarray]], masks: np.ndarray, final: bool
) -> list[np.ndarray]:
    """Every effect ``K_m^dagger K_m`` of a small circuit, by enumeration."""
    depth, n = masks.shape
    sites = [(t, i) for t in range(depth) for i in range(n) if masks[t, i]]
    finals = list(range(n)) if final else []
    effects = []
    for bits in itertools.product((0, 1), repeat=len(sites) + len(finals)):
        k = np.eye(2**n, dtype=complex)
        b = iter(bits)
        for t in range(depth):
            k = _layer_unitary(gates[t], n) @ k
            for i in range(n):
                if masks[t, i]:
                    k = _projector(n, i, next(b)) @ k
        for i in finals:
            k = _projector(n, i, next(b)) @ k
        e = k.conj().T @ k
        if np.trace(e).real > 1e-12:
            effects.append(e)
    return effects


def _rho(theta: np.ndarray) -> np.ndarray:
    psi = product_state(theta)
    return np.outer(psi, psi.conj())


def _exact_f(effects: list[np.ndarray]) -> np.ndarray:
    """``F = sum_m Pr(m) v v^T`` for the mixed input, constant excluded."""
    dim = effects[0].shape[0]
    f = 0.0
    for e in effects:
        prob = np.trace(e).real / dim
        v = conditional_means(e / np.trace(e).real)[1:]
        f = f + prob * np.outer(v, v)
    return np.asarray(f)


def test_circle_weights_are_the_second_moments() -> None:
    """``E[B_a B_b] = 2^{-|a|} delta_ab`` for angles uniform on the circle."""
    theta = np.random.default_rng(0).uniform(0, 2 * np.pi, (200_000, 2))
    b = bloch_monomials(theta)
    gram = b.T @ b / theta.shape[0]
    np.testing.assert_allclose(gram, np.diag(circle_weights(2)), atol=1e-2)


def test_conditional_means_reproduce_the_born_probability() -> None:
    """``Tr[E rho(theta)] = c_0 sum_a v_a B_a / sqrt(w_a)``."""
    n = 3
    rng = np.random.default_rng(1)
    gates = draw_haar_circuit(n, 2, rng)
    masks = rng.random((2, n)) < 0.5
    for final in (False, True):
        e = sample_effect(gates, masks, final, rng)
        v = conditional_means(e)
        assert v[0] == pytest.approx(1.0)
        for _ in range(5):
            theta = rng.uniform(0, 2 * np.pi, n)
            exact = np.trace(e @ _rho(theta)).real
            pred = (bloch_monomials(theta) @ (v / np.sqrt(circle_weights(n)))) / 2**n
            assert pred == pytest.approx(exact, abs=1e-12)


def test_full_record_effect_has_rank_one() -> None:
    rng = np.random.default_rng(2)
    gates = draw_haar_circuit(3, 2, rng)
    e = sample_effect(gates, np.ones((2, 3), bool), True, rng)
    assert np.linalg.matrix_rank(e, tol=1e-9) == 1
    assert np.trace(e).real == pytest.approx(1.0)


@pytest.mark.parametrize("final", [False, True])
def test_sampled_spectrum_matches_enumeration(final: bool) -> None:
    """Born-drawn records estimate the exact ``F``; ``mu`` in ``[0, 1]``."""
    n, depth = 2, 2
    rng = np.random.default_rng(3)
    gates = draw_haar_circuit(n, depth, rng)
    masks = np.array([[True, False], [False, True]])
    f = _exact_f(_exact_effects(gates, masks, final))
    mu_exact = np.linalg.eigvalsh(f)
    assert mu_exact.min() > -1e-12 and mu_exact.max() < 1 + 1e-12
    v = np.array(
        [
            conditional_means(sample_effect(gates, masks, final, rng))
            for _ in range(6000)
        ]
    )
    spec = record_spectrum(v)
    np.testing.assert_allclose(np.sort(spec.mu)[-3:], np.sort(mu_exact)[-3:], atol=0.03)


def test_spectral_r2_is_consistent() -> None:
    """``R2(1)`` spectral equals the direct estimate; ``R2(K)`` grows with ``K``."""
    n = 3
    rng = np.random.default_rng(4)
    gates = draw_haar_circuit(n, 3, rng)
    masks = rng.random((3, n)) < 0.4
    v = np.array(
        [conditional_means(sample_effect(gates, masks, True, rng)) for _ in range(500)]
    )
    spec = record_spectrum(v)
    beta = degree_target_circle(n, 1, rng)
    r1 = spectral_r2(spec, beta, 1)
    assert r1 == pytest.approx(single_shot_r2(v, beta), rel=1e-10)
    values = [spectral_r2(spec, beta, k) for k in (1, 4, 16, 256, np.inf)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:], strict=False))
    assert values[-1] <= 1 + 1e-9
    assert n_learn(spec.mu, 1) <= n_learn(spec.mu, 16) <= n_learn(spec.mu, np.inf)


def test_effect_span_and_approximation_bias() -> None:
    """The span is that of the enumerated effects; ``E_perp`` limits cases."""
    n, depth = 3, 2
    rng = np.random.default_rng(5)
    gates = draw_haar_circuit(n, depth, rng)
    masks = np.array([[True, False, True], [False, True, False]])
    for final in (False, True):
        span = effect_span(gates, masks, final)
        f = _exact_f(_exact_effects(gates, masks, final))
        lam, vec = np.linalg.eigh(f)
        exact = vec[:, lam > 1e-10]
        for k in (1, 2, 3):
            beta = degree_target_circle(n, k, rng)
            coords = (np.sqrt(circle_weights(n)) * beta)[1:]
            inside = float(np.sum((exact.T @ coords) ** 2) / (coords @ coords))
            assert approximation_bias(span, beta) == pytest.approx(1 - inside, abs=1e-8)
    # No mid-circuit measurement and no final readout: nothing is seen.
    span0 = effect_span(gates, np.zeros((depth, n), bool), False)
    assert approximation_bias(span0, single_site_target(n, 1, "z")) == pytest.approx(1)
    # p = 1 in the first layer: the Z part of the inputs after one layer is
    # read, but the targets stay partly out of reach.
    beta = single_site_target(n, 0, "x")
    assert 0 <= approximation_bias(effect_span(gates, masks, True), beta) <= 1


def test_targets_have_unit_variance_on_the_circle() -> None:
    rng = np.random.default_rng(6)
    theta = rng.uniform(0, 2 * np.pi, (100_000, 3))
    for beta in (
        degree_target_circle(3, 2, rng),
        single_site_target(3, 1, "x"),
        single_site_target(3, 2, "z"),
    ):
        y = target_values(beta, theta)
        assert y.mean() == pytest.approx(0.0, abs=2e-2)
        assert y.var() == pytest.approx(1.0, rel=3e-2)


def test_sample_shots_means_match_the_transfer_matrix() -> None:
    """The batched sampler has the means ``C(p) B(theta)`` of the exact model."""
    n, depth, p = 3, 2, 0.4
    rng = np.random.default_rng(7)
    layers = layers_from_gates(draw_haar_circuit(n, depth, rng), n)
    theta = rng.uniform(0, 2 * np.pi, (2, n))
    shots = sample_shots(layers, n, p, theta, 20_000, rng, batch=7000)
    assert shots.shape == (2, 20_000, depth * n + n)
    assert set(np.unique(shots)).issubset({-1, 0, 1})
    exact = bloch_monomials(theta) @ transfer_matrix(layers, n, p).T
    np.testing.assert_allclose(shots.mean(axis=1), exact, atol=0.03)


def test_shot_groups_average_disjoint_groups() -> None:
    shots = np.arange(2 * 8 * 3).reshape(2, 8, 3).astype(np.int8)
    g = shot_groups(shots, 4, 10)
    assert g.shape == (2, 2, 3)
    np.testing.assert_allclose(g[1, 0], shots[0, 4:8].mean(axis=0))


def test_generalized_spectrum_matches_scipy() -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(size=(5, 3))
    signal = a @ a.T
    b = rng.normal(size=(5, 5))
    noise = b @ b.T
    ours = generalized_spectrum(signal, noise)
    ref = scipy.linalg.eigh(signal, signal + noise, eigvals_only=True)[::-1]
    np.testing.assert_allclose(ours, ref, atol=1e-10)
    assert ours.max() <= 1 and ours.min() >= 0


def test_ridge_sweep_bias_and_variance() -> None:
    """No noise: zero error at small lambda; noise: variance > 0, sum holds."""
    rng = np.random.default_rng(9)
    w = rng.normal(size=4)
    x = rng.normal(size=(3, 300, 4))
    y = x[0] @ w
    lam = np.logspace(-8, 1, 10)
    clean = ridge_sweep(x[:1], y, x[1:2], x[1] @ w, x[2:3], x[2] @ w, lam)
    assert clean.test_mse[0] < 1e-10 and clean.variance[0] == 0
    noisy_train = x[0][None] + 0.3 * rng.normal(size=(5, 300, 4))
    noisy_test = x[2][None] + 0.3 * rng.normal(size=(5, 300, 4))
    noisy_val = x[1][None] + 0.3 * rng.normal(size=(5, 300, 4))
    sweep = ridge_sweep(noisy_train, y, noisy_val, x[1] @ w, noisy_test, x[2] @ w, lam)
    assert np.all(sweep.variance > 0)
    np.testing.assert_allclose(
        sweep.test_mse, sweep.bias2 + sweep.variance * 4 / 5, rtol=1e-10
    )
    # Strong regularisation shrinks to the mean: the error tends to Var(y).
    assert sweep.test_mse[-1] > sweep.test_mse[0]


def _load_script(name: str) -> ModuleType:
    """Import a script of experiments/ridge_spectrum, which is not a package."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "ridge_spectrum" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_script_and_plots_run(tmp_path: Path) -> None:
    """Both levels run on a tiny case, and the plot script reads them."""
    script = _load_script("ridge_spectrum.py")
    plots = _load_script("plot_ridge_spectrum.py")
    for seed in (0, 1):
        one = script.run_level1(3, seed, [0.1, 0.4], 200)
        two = script.run_level2(3, seed, [0.1, 0.4], 8, (60, 30, 30), 4)
        (tmp_path / f"level1_N3_seed{seed}.json").write_text(json.dumps(one))
        (tmp_path / f"level2_N3_seed{seed}.json").write_text(json.dumps(two))
    row = one["per_p"][1]["full"]
    assert len(row["n_learn"]) == len(one["K"])
    assert 0 <= row["targets"]["k1"]["e_perp"] <= 1
    ridge = two["per_p"][0]["targets"]["k1"]["ridge"]
    assert set(ridge) == {"1", "4", "inf"}
    assert ridge["inf"]["variance"] == 0
    runs = plots.load(tmp_path)
    out = tmp_path / "figures"
    out.mkdir()
    summary: dict = {}
    for fig in (plots.fig_nlearn, plots.fig_level2_bias, plots.fig_optimum):
        fig(runs, out, summary)
    assert (out / "n_learn.png").is_file() and "fig6" in summary
