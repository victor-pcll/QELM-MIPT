"""Scores of the entropy-prediction task, QELM.evaluation.entropy_regression (#230)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.entropy_regression import (
    even_features,
    one_hot,
    phase_labels,
    r2_within,
    stratified_split,
)


def test_stratified_split_keeps_the_share_of_every_group() -> None:
    groups = np.repeat([0.0, 0.2, 0.5], [10, 20, 7])
    train, test = stratified_split(groups, 0.7, np.random.default_rng(0))
    assert np.intersect1d(train, test).size == 0
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(37))
    for level, n in ((0.0, 10), (0.2, 20), (0.5, 7)):
        assert np.sum(groups[train] == level) == round(0.7 * n)
        assert np.sum(groups[test] == level) >= 1


def test_stratified_split_is_reproducible() -> None:
    groups = np.repeat([0.0, 1.0], 10)
    a = stratified_split(groups, 0.5, np.random.default_rng(3))
    b = stratified_split(groups, 0.5, np.random.default_rng(3))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
def test_stratified_split_rejects_a_bad_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="train_fraction"):
        stratified_split(np.zeros(4), fraction, np.random.default_rng(0))


def test_stratified_split_rejects_a_group_of_one() -> None:
    with pytest.raises(ValueError, match="need >= 2"):
        stratified_split(np.array([0.0, 0.0, 1.0]), 0.5, np.random.default_rng(0))


def test_one_hot() -> None:
    out = one_hot(np.array([0.1, 0.3, 0.1]), [0.1, 0.3])
    np.testing.assert_array_equal(out, [[1, 0], [0, 1], [1, 0]])
    with pytest.raises(ValueError, match="exactly one"):
        one_hot(np.array([0.2]), [0.1, 0.3])


def test_even_features_appends_the_squares() -> None:
    X = np.array([[1.0, -1.0, 0.0], [0.5, 0.0, -0.5]])
    out = even_features(X)
    assert out.shape == (2, 6)
    np.testing.assert_array_equal(out[:, 3:], X**2)
    # Flipping every sign leaves the squares unchanged.
    np.testing.assert_array_equal(even_features(-X)[:, 3:], out[:, 3:])


def test_phase_labels() -> None:
    np.testing.assert_array_equal(
        phase_labels(np.array([0.0, 0.16, 0.17, 0.5]), 0.17), [1, 1, -1, -1]
    )


def test_r2_within_of_perfect_and_group_mean_predictions() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat([0.0, 0.5], 50)
    y = groups * 4 + rng.normal(size=100)
    pooled, per = r2_within(y, y, groups)
    assert pooled == pytest.approx(1.0)
    assert per == {0.0: pytest.approx(1.0), 0.5: pytest.approx(1.0)}
    # The mean of each group, the best function of the group alone: 0.
    means = np.where(groups == 0, y[:50].mean(), y[50:].mean())
    pooled, per = r2_within(y, means, groups)
    assert pooled == pytest.approx(0.0, abs=1e-12)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in per.values())


def test_r2_within_skips_a_group_without_spread() -> None:
    groups = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([1.0, 3.0, 0.0, 0.0])
    pred = np.array([1.0, 3.0, 5.0, 5.0])
    pooled, per = r2_within(y, pred, groups)
    assert np.isnan(per[1.0])
    assert per[0.0] == pytest.approx(1.0)
    # The badly predicted group has no spread: it does not enter the pool.
    assert pooled == pytest.approx(1.0)
    pooled, _ = r2_within(np.zeros(2), np.ones(2), np.zeros(2))
    assert np.isnan(pooled)
