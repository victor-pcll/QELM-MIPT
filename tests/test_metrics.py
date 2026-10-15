"""Métriques de régression : MSE, NMSE, R2 (issue #28).

Les valeurs sont comparées à scikit-learn (``mean_squared_error``,
``r2_score``) et à des calculs numpy directs.
"""

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score

from QELM.evaluation.metrics import MSE, NMSE, R2, _as_pair

CASES = [
    ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]),  # exemple de la doc sklearn
    ([1, 2, 3], [1.0, 2.5, 2.0]),  # entiers et flottants mélangés
    ([0.1, -4.2, 3.3, 8, -1], [0.0, -4.0, 3.0, 7.5, -2.0]),
    (list(np.linspace(-1, 1, 50)), list(np.sin(np.linspace(-1, 1, 50)))),
]


@pytest.mark.parametrize(("y_true", "y_pred"), CASES)
def test_mse_matches_sklearn(y_true, y_pred) -> None:
    assert MSE(y_true, y_pred) == pytest.approx(mean_squared_error(y_true, y_pred))


@pytest.mark.parametrize(("y_true", "y_pred"), CASES)
def test_R2_matches_sklearn(y_true, y_pred) -> None:
    assert R2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


@pytest.mark.parametrize(("y_true", "y_pred"), CASES)
def test_nmse_matches_numpy(y_true, y_pred) -> None:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    expected = np.mean((yt - yp) ** 2) / np.mean(yt**2)
    assert NMSE(y_true, y_pred) == pytest.approx(expected)


def test_known_values_accept_lists() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    assert MSE(y_true, y_pred) == pytest.approx(1 / 3)
    assert NMSE(y_true, y_pred) == pytest.approx((1 / 3) / (14 / 3))
    assert R2(y_true, y_pred) == pytest.approx(0.5)


def test_perfect_prediction() -> None:
    y = np.array([1.0, -2.0, 3.5])
    assert MSE(y, y) == 0.0
    assert NMSE(y, y) == 0.0
    assert R2(y, y) == 1.0


def test_R2_zero_variance() -> None:
    """y_true constant : 1.0 si parfait, 0.0 sinon (comme r2_score)."""
    y_true = np.array([2.0, 2.0, 2.0])
    perfect = np.array([2.0, 2.0, 2.0])
    bad = np.array([1.0, 2.0, 3.0])
    assert R2(y_true, perfect) == 1.0
    assert R2(y_true, bad) == 0.0
    assert R2(y_true, perfect) == r2_score(y_true, perfect)
    assert R2(y_true, bad) == r2_score(y_true, bad)


def test_R2_can_be_negative() -> None:
    """Pire que prédire la moyenne : R2 < 0, comme sklearn."""
    y_true, y_pred = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
    assert R2(y_true, y_pred) == pytest.approx(-3.0)
    assert R2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


def test_nmse_all_zero_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        NMSE([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("metric", [MSE, NMSE, R2])
def test_validation_applies_to_every_metric(metric) -> None:
    with pytest.raises(ValueError, match="empty"):
        metric([], [])
    with pytest.raises(ValueError, match="1-dimensional"):
        metric([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    with pytest.raises(
        ValueError, match=r"Shapes differ: y_true \(3,\), y_pred \(2,\)"
    ):
        metric([1, 2, 3], [1, 2])


def test_as_pair_returns_float_arrays() -> None:
    yt, yp = _as_pair([1, 2], [3, 4])
    assert yt.dtype == yp.dtype == np.float64
    assert yt.shape == yp.shape == (2,)


def test_as_pair_rejects_scalars() -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        _as_pair(1.0, 2.0)
