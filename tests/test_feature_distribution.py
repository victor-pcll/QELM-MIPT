"""Distribution of the linear observables of the unitary QELM (issue #191)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from QELM import load_config
from QELM.evaluation.feature_distribution import (
    convergence_table,
    distribution_moments,
    ks_distance,
    ks_to_normal,
    linear_observables,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_Z" / "exp_QELM_Z.py"


def test_linear_observables_splits_the_features_label_by_label() -> None:
    """Column ``o * Ns + k`` is ``<O_k>`` for the o-th label; the bias goes."""
    L, Ns, labels = 5, 3, ["Z", "Y", "X"]
    X = np.arange(L * len(labels) * Ns, dtype=float).reshape(L, -1)
    X = np.hstack([X, np.ones((L, 1))])
    blocks = linear_observables(X, labels, Ns)
    assert list(blocks) == labels
    for o, label in enumerate(labels):
        assert blocks[label].shape == (L, Ns)
        np.testing.assert_array_equal(blocks[label], X[:, o * Ns : (o + 1) * Ns])
    no_bias = linear_observables(X[:, :-1], labels, Ns, bias=False)
    np.testing.assert_array_equal(no_bias["X"], blocks["X"])


def test_linear_observables_checks_the_shape_and_the_bias() -> None:
    X = np.ones((4, 7))
    with pytest.raises(ValueError, match="shape"):
        linear_observables(X, ["Z", "Y"], 2)
    X[0, -1] = 0.5
    with pytest.raises(ValueError, match="bias"):
        linear_observables(X, ["Z", "Y"], 3)


def test_moments_of_known_laws() -> None:
    rng = np.random.default_rng(0)
    gauss = distribution_moments(rng.normal(1.0, 2.0, size=200_000))
    assert gauss["mean"] == pytest.approx(1.0, abs=0.02)
    assert gauss["std"] == pytest.approx(2.0, rel=0.01)
    assert gauss["skewness"] == pytest.approx(0.0, abs=0.02)
    assert gauss["excess_kurtosis"] == pytest.approx(0.0, abs=0.05)
    assert gauss["sem"] == pytest.approx(2.0 / np.sqrt(200_000), rel=0.01)
    uniform = distribution_moments(rng.uniform(-1, 1, size=200_000))
    assert uniform["excess_kurtosis"] == pytest.approx(-1.2, abs=0.02)
    assert -1 <= uniform["min"] < uniform["max"] <= 1


def test_a_constant_sample_has_no_shape() -> None:
    """A feature fixed by a symmetry: std 0, no skewness, no KS distance."""
    m = distribution_moments(np.zeros(10))
    assert m["std"] == 0.0 and m["mean"] == 0.0
    assert np.isnan(m["skewness"]) and np.isnan(m["excess_kurtosis"])
    assert np.isnan(ks_to_normal(np.zeros(10)))


def test_samples_must_be_1d_with_two_values() -> None:
    with pytest.raises(ValueError, match="1D"):
        distribution_moments([1.0])
    with pytest.raises(ValueError, match="1D"):
        distribution_moments(np.ones((3, 2)))


def test_ks_distances() -> None:
    rng = np.random.default_rng(3)
    gauss = rng.normal(size=5000)
    assert ks_to_normal(gauss) < 0.03
    assert ks_to_normal(rng.uniform(-1, 1, size=5000)) > 0.04
    assert ks_distance(gauss, gauss) == 0.0
    assert ks_distance(gauss, gauss + 10) == 1.0


def test_convergence_table_shrinks_like_one_over_sqrt_n() -> None:
    """Independent samples of one law: the KS distance to the largest one is
    of the order of its bound and decreases with n."""
    rng = np.random.default_rng(4)
    samples = {n: rng.beta(2, 5, size=n) for n in (5000, 100, 1000)}
    rows = convergence_table(samples)
    assert [r["n_samples"] for r in rows] == [100, 1000, 5000]
    assert rows[-1]["ks_to_reference"] == 0.0
    assert rows[0]["ks_to_reference"] > rows[1]["ks_to_reference"]
    for r in rows[:-1]:
        assert r["ks_to_reference"] < 1.63 * r["ks_bound"]  # 1 % critical value
    assert rows[0]["ks_bound"] == pytest.approx(np.sqrt(5100 / 500_000))
    external = convergence_table(samples, reference=rng.beta(2, 5, size=2000))
    assert external[-1]["ks_to_reference"] > 0.0
    with pytest.raises(ValueError, match="empty"):
        convergence_table({})


def test_unitary_qelm_has_no_linear_x_or_y_feature() -> None:
    """H conserves the Z parity and the initial state has a definite parity
    (both branches carry Ns // 2 excitations in total), so <X_k> = <Y_k> = 0
    exactly; <Z_k> varies from sample to sample."""
    spec = importlib.util.spec_from_file_location("exp_QELM_Z", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for h in (10.0, 1.0):
        cfg = load_config(["models=TFIM", "N=3", f"TFIM.h={h}", "samples.L=12"])
        out = module.run(cfg)
        blocks = linear_observables(out["X"], ["Z", "Y", "X"], 3)
        np.testing.assert_allclose(blocks["X"], 0.0, atol=1e-10)
        np.testing.assert_allclose(blocks["Y"], 0.0, atol=1e-10)
        assert np.all(np.std(blocks["Z"], axis=0) > 1e-3)
