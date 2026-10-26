"""Double-descent sweep and its features, ``QELM.evaluation.double_descent``."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.embedding import RotationEmbedding
from QELM.evaluation.double_descent import (
    add_bias,
    classification_error,
    double_descent_sweep,
    expectations,
    full_pool_fit,
    mse,
    n_train_grid,
    numerical_rank,
    observable_operators,
    one_hot,
    predict_classes,
    product_state_features,
    random_features,
)
from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.training import DoubleObservable, SingleObservable


def test_operators_follow_the_order_of_the_readouts() -> None:
    names, ops = observable_operators(3, ["Z", "X"], [["X", "X"], ["Z", "Y"]])
    assert names[:6] == ["Z_0", "Z_1", "Z_2", "X_0", "X_1", "X_2"]
    assert names[6:] == ["XX_0_1", "XX_0_2", "XX_1_2", "ZY_0_1", "ZY_0_2", "ZY_1_2"]
    dense = SingleObservable({"observables": ["Z", "X"]}).operators(3)
    dense += DoubleObservable({"pairs": [["X", "X"], ["Z", "Y"]]}).operators(3)
    assert len(ops) == len(dense) == 12
    for sparse, full in zip(ops, dense, strict=True):
        np.testing.assert_allclose(sparse.toarray(), full)


def test_expectations_match_the_dense_formula() -> None:
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(5, 8)) + 1j * rng.normal(size=(5, 8))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    _, ops = observable_operators(3, ["X", "Y", "Z"], [["Z", "Z"]])
    values = expectations(psi, ops)
    for n in range(5):
        for k, op in enumerate(ops):
            expected = np.real(psi[n].conj() @ op.toarray() @ psi[n])
            assert values[n, k] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_t0_features_are_the_product_state_closed_form(axis: str) -> None:
    """Check 2 of the issue: no dynamics, the features are fixed functions."""
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, np.pi, size=(10, 4))
    emb = RotationEmbedding({"N": 4, "axis": axis})
    single, pairs = ["X", "Y", "Z"], [["X", "X"], ["Y", "Y"], ["Z", "Z"], ["X", "Z"]]
    _, ops = observable_operators(4, single, pairs)
    simulated = expectations(emb.states(theta), ops)
    closed = product_state_features(emb.bloch_vectors(theta), single, pairs)
    np.testing.assert_allclose(simulated, closed, atol=1e-12)


def test_t0_rank_drops_with_the_vanishing_observables() -> None:
    """R_y gives a real state: <Y_i> = <Y_i Y_j> = 0, so the rank is lower."""
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, np.pi, size=(400, 4))
    emb = RotationEmbedding({"N": 4})
    single, pairs = ["X", "Y", "Z"], [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
    _, ops = observable_operators(4, single, pairs)
    t0 = add_bias(expectations(emb.states(theta), ops))
    assert t0.shape[1] == 3 * 4 + 3 * 6 + 1 == 31
    # The 4 <Y_i> and 6 <Y_i Y_j> are 0: 21 columns left, all independent.
    assert numerical_rank(t0) == 21
    np.random.seed(0)
    model = TransverseFieldIsingModel({"N": 4, "J": 1.0, "h": 1.0})
    U = model.propagator(dt=10.0)
    quench = add_bias(expectations(emb.states(theta) @ U.T, ops))
    assert numerical_rank(quench) == 31


def test_random_features_with_anchors_have_full_rank() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(0, np.pi, size=(500, 4))
    anchored = random_features(x, 40, np.random.default_rng(0), anchors=x)
    free = random_features(x, 40, np.random.default_rng(0))
    assert anchored.shape == (500, 40)
    assert anchored.min() >= 0
    assert numerical_rank(add_bias(anchored)) == 41
    # Without anchors, always-on ReLUs are affine in x: the rank collapses.
    assert numerical_rank(add_bias(free)) < 41
    for activation in ("tanh", "cos"):
        out = random_features(x, 5, np.random.default_rng(0), activation=activation)
        assert np.all(np.abs(out) <= 1)
    with pytest.raises(ValueError, match="activation"):
        random_features(x, 5, rng, activation="sigmoid")


def test_n_train_grid() -> None:
    np.testing.assert_array_equal(
        n_train_grid([0.01, 0.5, 1.0, 2.0], 25), [1, 12, 25, 50]
    )


def test_one_hot_prediction_and_errors() -> None:
    Y = one_hot(np.array([0, 2, 1]), 3)
    np.testing.assert_array_equal(Y, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    outputs = np.array([[0.9, 0.1, 0.0], [0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
    np.testing.assert_array_equal(predict_classes(outputs), [0, 1, 1])
    assert classification_error(np.array([0, 2, 1]), outputs) == pytest.approx(1 / 3)
    assert mse(Y, Y) == 0.0
    assert mse(Y, np.zeros((3, 3))) == pytest.approx(1 / 3)


def _linear_task(
    n: int, d: int, n_classes: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian features; the class is the largest of noisy linear scores."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n, d))
    scores = phi @ rng.normal(size=(d, n_classes))
    labels = np.argmax(scores + 0.5 * rng.normal(size=scores.shape), axis=1)
    return phi, labels


@pytest.mark.parametrize("n_classes", [2, 4])
def test_sweep_peaks_at_the_interpolation_threshold(n_classes: int) -> None:
    """Check 3 of the issue, on Gaussian features of known rank."""
    phi, y = _linear_task(1500, 20, n_classes)
    ratios = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 5.0]
    sweep = double_descent_sweep(
        phi[:1000],
        y[:1000],
        phi[1000:],
        y[1000:],
        ratios,
        30,
        np.random.default_rng(0),
    )
    assert (sweep.n_features, sweep.n_columns) == (20, 20)
    np.testing.assert_array_equal(sweep.n_train, [5, 10, 15, 18, 20, 22, 30, 40, 100])
    assert sweep.test_mse.shape == (9, 30)
    assert sweep.peak_alpha() == 1.0
    # Below the threshold the readout interpolates: train MSE 0.
    below = sweep.alpha < 1
    np.testing.assert_allclose(sweep.train_mse[below], 0.0, atol=1e-18)
    np.testing.assert_allclose(sweep.train_error[below], 0.0)
    # Second descent: the test MSE falls again far above the threshold.
    median = np.median(sweep.test_mse, axis=1)
    assert median[-1] < median[4] / 10
    # The weights blow up at the threshold.
    norms = np.median(sweep.weight_norm, axis=1)
    assert np.argmax(norms) == 4


def test_ridge_penalty_smooths_the_peak() -> None:
    phi, y = _linear_task(1500, 20)
    args = (phi[:1000], y[:1000], phi[1000:], y[1000:], [0.5, 1.0, 2.0], 20)
    free = double_descent_sweep(*args, np.random.default_rng(0))
    ridge = double_descent_sweep(*args, np.random.default_rng(0), ridge_alpha=1.0)
    at_one = 1
    assert np.median(ridge.test_mse[at_one]) < np.median(free.test_mse[at_one]) / 5
    assert np.median(ridge.weight_norm[at_one]) < np.median(free.weight_norm[at_one])


def test_peak_follows_the_rank_not_the_column_count() -> None:
    """The diagnostic note: dependent columns move the threshold."""
    phi, y = _linear_task(1500, 12)
    redundant = np.hstack([phi, phi[:, :4] + phi[:, 4:8]])  # 16 columns, rank 12
    sweep = double_descent_sweep(
        redundant[:1000],
        y[:1000],
        redundant[1000:],
        y[1000:],
        [0.5, 1.0, 16 / 12, 2.0],
        30,
        np.random.default_rng(0),
    )
    assert (sweep.n_features, sweep.n_columns) == (12, 16)
    assert sweep.n_train[1] == 12
    assert sweep.peak_alpha() == 1.0


def test_sweep_needs_a_large_enough_pool() -> None:
    phi, y = _linear_task(30, 5)
    with pytest.raises(ValueError, match="exceeds the pool"):
        double_descent_sweep(phi, y, phi, y, [10.0], 2, np.random.default_rng(0))


def test_full_pool_fit_and_confusion_matrix() -> None:
    phi, y = _linear_task(900, 6, n_classes=3)
    accuracy, confusion = full_pool_fit(phi[:600], y[:600], phi[600:], y[600:], 3)
    assert accuracy > 0.7
    assert confusion.shape == (3, 3)
    # Rows are the true classes, so they count the test samples of each.
    np.testing.assert_array_equal(
        confusion.sum(axis=1), np.bincount(y[600:], minlength=3)
    )
    assert np.trace(confusion) / confusion.sum() == pytest.approx(accuracy)
    # A class absent from the test set still gets its row and column.
    _, padded = full_pool_fit(phi[:600], y[:600], phi[600:610], np.zeros(10, int), 4)
    assert padded.shape == (4, 4)
