"""The NumPy K-NN, ``QELM.evaluation.knn`` (issue #214)."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from QELM import load_config
from QELM.evaluation.knn import CustomKNN, default_n_neighbors

ALGORITHMS = ["auto", "brute", "kd_tree", "ball_tree"]
SCRIPT = Path(__file__).resolve().parents[1] / "experiments/QELM_Q/exp_QELM_Q.py"


def _sklearn(X: np.ndarray, y: np.ndarray, algorithm: str) -> KNeighborsClassifier:
    """The K-NN of the Bayes ceilings: K = round(sqrt(n)), inverse distance."""
    K = default_n_neighbors(len(y))
    model = KNeighborsClassifier(n_neighbors=K, weights="distance", algorithm=algorithm)
    return model.fit(X, y)


def _assert_same_as_sklearn(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, ties: str
) -> None:
    """Same neighbors, distances, labels and probabilities as every algorithm.

    The trees compute ``||x - x_i||`` from the differences, as CustomKNN does,
    so they agree to rounding. scikit-learn's brute force uses the expansion
    ``||x||^2 - 2 x.x_i + ||x_i||^2``, whose squared distance is off by about
    ``1e-16 ||x||^2``: for two very close points (``d ~ 1e-5`` in the QELM
    features) the distance, hence the weight ``1 / d``, is then off by up to
    ``1e-6`` relative. The labels must still be identical.
    """
    ours = CustomKNN(ties=ties).fit(X, y)
    dist, ind = ours.kneighbors(X_test)
    for algorithm in ALGORITHMS:
        ref = _sklearn(X, y, algorithm)
        ref_dist, ref_ind = ref.kneighbors(X_test)
        np.testing.assert_array_equal(ind, ref_ind)
        np.testing.assert_array_equal(ours.classes_, ref.classes_)
        np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
        proba, ref_proba = ours.predict_proba(X_test), ref.predict_proba(X_test)
        if ref._fit_method == "brute":
            np.testing.assert_allclose(dist**2, ref_dist**2, rtol=0, atol=1e-13)
            np.testing.assert_allclose(proba, ref_proba, rtol=0, atol=1e-5)
        else:
            np.testing.assert_allclose(dist, ref_dist, rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(proba, ref_proba, rtol=0, atol=1e-12)


def _tied_kth(X: np.ndarray, X_test: np.ndarray, K: int) -> np.ndarray:
    """Rows whose K-th and (K+1)-th squared distances are equal."""
    d2 = np.sort(((X_test[:, None, :] - X[None, :, :]) ** 2).sum(-1), axis=1)
    return d2[:, K - 1] == d2[:, K]


@pytest.mark.parametrize("ties", ["average", "index"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_matches_sklearn_on_continuous_features(ties: str, n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    X = rng.normal(size=(300, 5))
    y = rng.integers(0, n_classes, size=300)
    X_test = rng.normal(size=(200, 5))
    _assert_same_as_sklearn(X, y, X_test, ties)


def test_keeps_labels_that_are_not_integers() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = np.where(rng.random(50) < 0.5, "Q1", "Q0")
    model = CustomKNN().fit(X, y)
    np.testing.assert_array_equal(model.classes_, ["Q0", "Q1"])
    assert set(model.predict(X)) <= {"Q0", "Q1"}
    _assert_same_as_sklearn(X, y, rng.normal(size=(40, 2)), "average")


def test_decision_regions_match_sklearn_on_a_grid() -> None:
    """Two interleaved sectors; the regions on a 2-D grid are identical."""
    rng = np.random.default_rng(3)
    centers = np.array([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    labels = np.array([1, 0, 1, 0])
    which = rng.integers(0, 4, size=400)
    X = centers[which] + 0.5 * rng.normal(size=(400, 2))
    y = labels[which]
    gx, gy = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-2, 2, 41))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    _assert_same_as_sklearn(X, y, grid, "average")
    # The K-NN follows the four bundles, which a straight line cannot.
    regions = CustomKNN().fit(X, y).predict(centers)
    np.testing.assert_array_equal(regions, labels)


def _qelm_features(p: float) -> tuple[np.ndarray, np.ndarray, int]:
    """The <Z_k> and Q of a small run of exp_QELM_Q.py, without the bias."""
    spec = importlib.util.spec_from_file_location("exp_QELM_Q", SCRIPT)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = load_config(
        ["models=TFIM", "N=4", "samples.L=60", f"projective_measurement.p={p}"]
    )
    out = module.run(cfg)
    return out["X"][:, :-1], out["Y"], int(cfg.samples.train_size)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_matches_sklearn_on_the_qelm_sweep(p: float) -> None:
    X, Y, train = _qelm_features(p)
    X_train, y_train, X_test = X[:train], Y[:train], X[train:]
    K = default_n_neighbors(train)
    assert not _tied_kth(X_train, X_test, K).any()
    for ties in ("average", "index"):
        _assert_same_as_sklearn(X_train, y_train, X_test, ties)


@pytest.mark.parametrize(
    ("n_train", "K"), [(1, 1), (2, 1), (3, 2), (10, 3), (48, 7), (500, 22)]
)
def test_default_k_is_the_rounded_square_root(n_train: int, K: int) -> None:
    assert default_n_neighbors(n_train) == K
    X = np.arange(n_train, dtype=float)[:, None]
    assert CustomKNN().fit(X, np.zeros(n_train)).n_neighbors_ == K


def test_k_larger_than_the_training_set_is_reduced() -> None:
    X = np.arange(4, dtype=float)[:, None]
    model = CustomKNN(n_neighbors=10).fit(X, [0, 0, 1, 1])
    assert model.n_neighbors_ == 4
    assert model.kneighbors([[0.0]])[1].tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize("ties", ["average", "index"])
def test_a_point_on_a_training_point_takes_its_label(ties: str) -> None:
    X = np.array([[0.0], [0.1], [0.2], [5.0]])
    y = np.array([1, 0, 0, 0])
    model = CustomKNN(n_neighbors=3, ties=ties).fit(X, y)
    np.testing.assert_array_equal(model.predict_proba([[0.0]]), [[0.0, 1.0]])
    assert model.predict([[0.0]])[0] == 1
    ref = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
    np.testing.assert_array_equal(
        model.predict_proba([[0.0]]), ref.predict_proba([[0.0]])
    )


def test_points_on_x_that_fill_all_places_share_the_vote() -> None:
    """Three training points on x, K = 2: each counts 2/3 with ties="average"."""
    X = np.array([[1.0], [1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    average = CustomKNN(n_neighbors=2).fit(X, y)
    np.testing.assert_allclose(average.predict_proba([[1.0]]), [[2 / 3, 1 / 3]])
    index = CustomKNN(n_neighbors=2, ties="index").fit(X, y)
    np.testing.assert_array_equal(index.predict_proba([[1.0]]), [[1.0, 0.0]])


def test_average_is_the_mean_over_all_tie_breaks() -> None:
    """One point inside the K-th distance, four on it, two free places (K = 3).

    Each choice of two tied points is made the lowest-index one by reordering
    the training set; the average of those ``ties="index"`` votes is the
    ``ties="average"`` vote.
    """
    X = np.array([[0.5], [1.0], [-1.0], [1.0], [-1.0], [3.0]])
    y = np.array([0, 1, 1, 0, 1, 0])
    x = np.array([[0.0]])
    tied = [1, 2, 3, 4]
    votes = []
    for chosen in itertools.combinations(tied, 2):
        rest = [i for i in tied if i not in chosen]
        order = [0, *chosen, *rest, 5]
        model = CustomKNN(n_neighbors=3, ties="index").fit(X[order], y[order])
        votes.append(model.predict_proba(x)[0])
    average = CustomKNN(n_neighbors=3).fit(X, y).predict_proba(x)[0]
    np.testing.assert_allclose(average, np.mean(votes, axis=0), atol=1e-15)
    # By hand: 2 (from 0.5) + 2/4 (one tied point of label 0) for label 0.
    np.testing.assert_allclose(average, np.array([2.5, 1.5]) / 4.0)


def _hypercube(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n, d)), rng.integers(0, 2, size=n)


def test_average_does_not_depend_on_the_order_of_the_training_set() -> None:
    X, y = _hypercube(200, 4, seed=0)
    X_test, _ = _hypercube(100, 4, seed=1)
    perm = np.random.default_rng(2).permutation(200)
    first = CustomKNN().fit(X, y)
    second = CustomKNN().fit(X[perm], y[perm])
    np.testing.assert_allclose(
        first.predict_proba(X_test), second.predict_proba(X_test), atol=1e-14
    )
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))


def test_index_takes_the_lowest_indices_among_tied_points() -> None:
    X, y = _hypercube(200, 4, seed=3)
    X_test, _ = _hypercube(50, 4, seed=4)
    K = default_n_neighbors(200)
    dist, ind = CustomKNN(ties="index").fit(X, y).kneighbors(X_test)
    for row, x in enumerate(X_test):
        d2 = ((X - x) ** 2).sum(axis=1)
        expected = np.lexsort((np.arange(200), d2))[:K]
        np.testing.assert_array_equal(ind[row], expected)
    # The distances themselves are those of scikit-learn, whatever the ties.
    for algorithm in ALGORITHMS:
        ref_dist, _ = _sklearn(X, y, algorithm).kneighbors(X_test)
        np.testing.assert_array_equal(dist, ref_dist)


@pytest.mark.parametrize("ties", ["average", "index"])
def test_hypercube_rows_without_a_tie_match_sklearn(ties: str) -> None:
    """Where the K-th neighbor is not tied, both rules give scikit-learn's vote."""
    X, y = _hypercube(30, 6, seed=5)
    X_test, _ = _hypercube(64, 6, seed=6)
    free = ~_tied_kth(X, X_test, default_n_neighbors(30))
    assert free.any()
    ours = CustomKNN(ties=ties).fit(X, y)
    for algorithm in ALGORITHMS:
        ref = _sklearn(X, y, algorithm)
        np.testing.assert_allclose(
            ours.predict_proba(X_test[free]),
            ref.predict_proba(X_test[free]),
            atol=1e-12,
        )


def test_score_is_the_accuracy() -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    model = CustomKNN(n_neighbors=2).fit(X, y)
    assert model.score([[0.5], [10.5]], [0, 1]) == 1.0
    assert model.score([[0.5], [10.5]], [1, 1]) == 0.5


def test_errors() -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        CustomKNN(n_neighbors=0)
    with pytest.raises(ValueError, match="ties"):
        CustomKNN(ties="random")
    with pytest.raises(ValueError, match="n_train"):
        default_n_neighbors(0)
    with pytest.raises(RuntimeError, match="not fitted"):
        CustomKNN().predict([[0.0]])
    model = CustomKNN()
    with pytest.raises(ValueError, match="2-D"):
        model.fit(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="y must have shape"):
        model.fit(np.zeros((3, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="empty"):
        model.fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError, match="NaN"):
        model.fit(np.array([[np.nan, 0.0]]), np.zeros(1))
    model.fit(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="NaN"):
        model.predict(np.array([[np.inf, 0.0]]))
