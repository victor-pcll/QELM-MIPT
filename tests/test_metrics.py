"""Métriques de régression : MSE, NMSE, R2 (issue #28) et l'erreur standard
de MSE et NMSE (issue #89).

Les valeurs sont comparées à scikit-learn (``mean_squared_error``,
``r2_score``) et à des calculs numpy directs.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    recall_score,
)

from QELM.evaluation.metrics import (
    MSE,
    NMSE,
    R2,
    MSE_err,
    NMSE_err,
    _as_pair,
    accuracy,
    balanced_accuracy,
    majority_accuracy,
)

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


def test_R2_constant_up_to_rounding() -> None:
    """y_true constant à l'arrondi près (issue #103) : même réponse que le
    cas exact, et non une valeur de l'ordre de -1e32."""
    y_true = np.array([1.0, 1.0 - 1e-16, 1.0])
    assert np.sum((y_true - y_true.mean()) ** 2) > 0.0  # pas exactement constant
    assert R2(y_true, np.ones(3)) == 1.0
    assert R2(y_true, np.array([1.0, 0.5, 1.0])) == 0.0


def test_R2_small_but_real_variance_is_kept() -> None:
    """Une vraie variance, même petite, n'est pas prise pour une constante."""
    y_true = np.array([1.0, 1.0 + 1e-3, 1.0 - 1e-3])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert R2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


def test_R2_can_be_negative() -> None:
    """Pire que prédire la moyenne : R2 < 0, comme sklearn."""
    y_true, y_pred = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
    assert R2(y_true, y_pred) == pytest.approx(-3.0)
    assert R2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


def test_nmse_all_zero_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        NMSE([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("metric", [MSE, NMSE, R2, MSE_err, NMSE_err])
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


@pytest.mark.parametrize(("y_true", "y_pred"), CASES)
def test_accuracy_matches_sklearn_on_the_signs(y_true, y_pred) -> None:
    """Classe +1 si y >= 0, -1 sinon, pour la cible et la prédiction (#124)."""
    expected = accuracy_score(np.asarray(y_true) >= 0, np.asarray(y_pred) >= 0)
    assert accuracy(y_true, y_pred) == pytest.approx(expected)


def test_accuracy_known_values() -> None:
    y_true = [1.0, -1.0, 1.0, -1.0]
    assert accuracy(y_true, [0.9, -0.2, -0.1, 0.3]) == 0.5
    assert accuracy(y_true, y_true) == 1.0
    assert accuracy(y_true, [-y for y in y_true]) == 0.0
    # 0 tombe dans la classe +1 ; le seuil se déplace.
    assert accuracy([0.0, -1e-3], [1.0, -1.0]) == 1.0
    assert accuracy([0.4, 0.6], [0.6, 0.4], threshold=0.5) == 0.0


def test_accuracy_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        accuracy([1.0, 2.0], [1.0])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(("y_true", "y_pred"), CASES)
def test_balanced_accuracy_matches_sklearn_on_the_signs(y_true, y_pred) -> None:
    """Moyenne des rappels des signes présents dans y_true (#130), comme sklearn.

    La référence est le rappel moyen ``recall_score(average="macro")`` sur les
    classes de y_true, la définition de ``balanced_accuracy_score``. Celle-ci
    n'a pas de paramètre ``labels`` et émet un UserWarning quand y_true et
    y_pred n'ont qu'un signe, comme ``[1, 2, 3]`` dans CASES (issue #133).
    Toute alerte fait échouer ce test.
    """
    true_cls = np.asarray(y_true) >= 0
    pred_cls = np.asarray(y_pred) >= 0
    expected = recall_score(
        true_cls, pred_cls, labels=np.unique(true_cls), average="macro"
    )
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(expected)


def test_constant_prediction_scores_half_whatever_the_imbalance() -> None:
    """Issue #130 : 85 % de signes +, une prédiction constante positive a une
    accuracy de 0.85 (la part majoritaire) mais une accuracy équilibrée de
    0.5."""
    y_true = np.array([0.5] * 85 + [-0.3] * 15)
    y_pred = np.full(100, 0.42)
    assert accuracy(y_true, y_pred) == pytest.approx(0.85)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)
    assert balanced_accuracy(y_true, -y_pred) == pytest.approx(0.5)
    assert balanced_accuracy(y_true, y_true) == 1.0
    assert balanced_accuracy(y_true, -y_true) == 0.0


def test_majority_accuracy_known_values() -> None:
    """Le classifieur du signe majoritaire de l'entraînement (#130)."""
    y_train = [1.0, 0.5, -1.0]
    assert majority_accuracy(y_train, [0.2, 0.3, -0.1, 0.4]) == 0.75
    assert majority_accuracy([-1.0, -0.5, 1.0], [0.2, 0.3, -0.1, 0.4]) == 0.25
    # Égalité : la classe +1 ; le seuil se déplace comme pour accuracy.
    assert majority_accuracy([1.0, -1.0], [1.0, -1.0]) == 0.5
    assert majority_accuracy([0.6, 0.7, 0.1], [0.55, 0.2], threshold=0.5) == 0.5


def test_majority_accuracy_rejects_empty_or_2d() -> None:
    with pytest.raises(ValueError, match="y_train"):
        majority_accuracy([], [1.0])
    with pytest.raises(ValueError, match="y_test"):
        majority_accuracy([1.0], [[1.0]])


@pytest.mark.parametrize(("y_true", "y_pred"), CASES)
def test_mse_err_is_the_sem_of_the_squared_errors(y_true, y_pred) -> None:
    """SEM = std(e, ddof=1) / sqrt(n) avec e_i = (y_i - yhat_i)**2 (#89)."""
    e = (np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2
    expected = np.std(e, ddof=1) / np.sqrt(e.size)
    assert MSE_err(y_true, y_pred) == pytest.approx(expected)


def test_nmse_err_reduces_to_mse_err_for_a_pm1_target() -> None:
    """Cible +-1 : la normalisation ne fluctue pas, SEM = MSE_err / 1."""
    rng = np.random.default_rng(0)
    y_true = rng.choice([-1.0, 1.0], size=200)
    y_pred = y_true + rng.normal(0.0, 0.3, size=200)
    assert NMSE_err(y_true, y_pred) == pytest.approx(MSE_err(y_true, y_pred))


def test_nmse_err_scales_like_nmse() -> None:
    """Multiplier y_true et y_pred par c ne change ni NMSE ni son erreur."""
    y_true, y_pred = map(np.asarray, CASES[2])
    assert NMSE_err(3.0 * y_true, 3.0 * y_pred) == pytest.approx(
        NMSE_err(y_true, y_pred)
    )


def test_errors_of_a_perfect_prediction_are_zero() -> None:
    y = np.array([1.0, -2.0, 3.0])
    assert MSE_err(y, y) == 0.0
    assert NMSE_err(y, y) == 0.0


def test_errors_of_a_single_sample_are_nan() -> None:
    """L'écart type d'un seul échantillon (ddof=1) n'est pas défini."""
    assert np.isnan(MSE_err([1.0], [0.5]))
    assert np.isnan(NMSE_err([1.0], [0.5]))


def test_nmse_err_rejects_an_all_zero_target() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        NMSE_err([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize(("metric", "metric_err"), [(MSE, MSE_err), (NMSE, NMSE_err)])
def test_error_matches_the_spread_over_independent_test_sets(
    metric, metric_err
) -> None:
    """L'erreur d'un seul jeu de test prédit l'écart type de la métrique sur
    des jeux de test indépendants de même taille (Monte Carlo, seed fixée).

    La cible y = x a une amplitude qui varie, donc la normalisation de NMSE
    fluctue aussi : c'est le cas où la méthode delta compte.
    """
    rng = np.random.default_rng(1234)
    n, repeats = 400, 2000
    values, errors = [], []
    for _ in range(repeats):
        y_true = rng.normal(0.0, 1.0, size=n)
        y_pred = 0.8 * y_true + rng.normal(0.0, 0.3, size=n)
        values.append(metric(y_true, y_pred))
        errors.append(metric_err(y_true, y_pred))
    spread = np.std(values, ddof=1)
    assert np.mean(errors) == pytest.approx(spread, rel=0.06)
