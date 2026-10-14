import numpy as np
import pytest

from QELM.evaluation.metrics import MSE, NMSE, R


def test_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert MSE(y, y) == 0.0
    assert NMSE(y, y) == 0.0
    assert R(y, y) == 1.0


def test_known_values_accept_lists() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    assert MSE(y_true, y_pred) == pytest.approx(1 / 3)
    assert NMSE(y_true, y_pred) == pytest.approx((1 / 3) / (14 / 3))
    assert R(y_true, y_pred) == pytest.approx(0.5)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shapes differ"):
        MSE([1.0, 2.0], [1.0])
