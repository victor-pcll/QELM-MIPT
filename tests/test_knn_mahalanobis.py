"""The Mahalanobis metric of the NumPy K-NN, ``QELM.evaluation.knn`` (#213)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from QELM import load_config
from QELM.evaluation.knn import CustomKNN

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/QELM_Q/exp_QELM_Q.py"


def _correlated(n: int, seed: int, d: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Two classes of strongly correlated, anisotropic Gaussian features."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) * np.array([10.0, 3.0, 1.0, 0.1])[:d]
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, d)) @ A + 0.5 * y[:, None]
    return X, y


@pytest.mark.parametrize("ties", ["average", "index"])
@pytest.mark.parametrize("algorithm", ["brute", "ball_tree"])
def test_matches_sklearn_mahalanobis(ties: str, algorithm: str) -> None:
    """Same neighbors, distances, labels and votes as scikit-learn's metric."""
    X, y = _correlated(300, seed=0)
    X_test, _ = _correlated(150, seed=1)
    ours = CustomKNN(ties=ties, metric="mahalanobis").fit(X, y)
    VI = np.linalg.inv(np.cov(X, rowvar=False))
    ref = KNeighborsClassifier(
        n_neighbors=ours.n_neighbors_,
        weights="distance",
        metric="mahalanobis",
        metric_params={"VI": VI},
        algorithm=algorithm,
    ).fit(X, y)
    dist, ind = ours.kneighbors(X_test)
    ref_dist, ref_ind = ref.kneighbors(X_test)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-10)
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(
        ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-10
    )


def test_covariance_and_precision() -> None:
    """Sigma is the global covariance, Sigma^+ its pseudo-inverse."""
    X, y = _correlated(200, seed=2)
    model = CustomKNN(metric="mahalanobis").fit(X, y)
    np.testing.assert_allclose(model.covariance_, np.cov(X, rowvar=False))
    np.testing.assert_allclose(
        model.precision_,
        np.linalg.pinv(model.covariance_, hermitian=True),
        rtol=1e-8,
        atol=1e-12,
    )
    assert model.rank_ == X.shape[1]


def test_distance_is_the_quadratic_form() -> None:
    """d(x, x_i) = sqrt((x - x_i)^T Sigma^+ (x - x_i)) for every neighbor."""
    X, y = _correlated(100, seed=3)
    X_test, _ = _correlated(20, seed=4)
    model = CustomKNN(metric="mahalanobis").fit(X, y)
    dist, ind = model.kneighbors(X_test)
    diff = X_test[:, None, :] - X[ind]
    expected = np.sqrt(np.einsum("nki,ij,nkj->nk", diff, model.precision_, diff))
    np.testing.assert_allclose(dist, expected, rtol=1e-10)
    # They are the K smallest of all the distances.
    all_diff = X_test[:, None, :] - X[None, :, :]
    d_all = np.einsum("nki,ij,nkj->nk", all_diff, model.precision_, all_diff)
    np.testing.assert_allclose(
        dist**2, np.sort(d_all, axis=1)[:, : model.n_neighbors_], rtol=1e-10
    )


def test_affine_invariance() -> None:
    """X -> X A + b changes the Mahalanobis votes only by rounding.

    Sigma becomes A^T Sigma A, so the distances do not change. A common factor
    on every feature (A = c 1) is the case of a rescaled Sigma (``ddof``).
    The Euclidean K-NN is not invariant.
    """
    X, y = _correlated(200, seed=5)
    X_test, _ = _correlated(100, seed=6)
    rng = np.random.default_rng(7)
    for A in (rng.normal(size=(4, 4)), 3.7 * np.eye(4)):
        b = rng.normal(size=4)
        ref = CustomKNN(metric="mahalanobis").fit(X, y)
        moved = CustomKNN(metric="mahalanobis").fit(X @ A + b, y)
        np.testing.assert_allclose(
            moved.predict_proba(X_test @ A + b), ref.predict_proba(X_test), atol=1e-8
        )
        np.testing.assert_allclose(
            moved.kneighbors(X_test @ A + b)[0], ref.kneighbors(X_test)[0], rtol=1e-8
        )
    # A rotation and stretch moves the Euclidean neighbors (a scale does not).
    A = rng.normal(size=(4, 4))
    euclid = CustomKNN().fit(X, y).predict_proba(X_test)
    moved_e = CustomKNN().fit(X @ A + b, y).predict_proba(X_test @ A + b)
    assert not np.allclose(euclid, moved_e)


def test_beats_euclidean_when_the_class_is_in_a_direction_of_small_spread() -> None:
    """A wide nuisance feature hides the class from the Euclidean distance."""
    rng = np.random.default_rng(8)
    n = 800
    y = rng.integers(0, 2, size=n)
    X = np.column_stack(
        [
            rng.normal(scale=100.0, size=n),
            np.where(y == 1, 0.5, -0.5) + 0.1 * rng.normal(size=n),
        ]
    )
    train = 600
    euclid = CustomKNN().fit(X[:train], y[:train]).score(X[train:], y[train:])
    maha = CustomKNN(metric="mahalanobis").fit(X[:train], y[:train])
    assert maha.score(X[train:], y[train:]) > 0.95
    assert euclid < 0.8


def test_singular_covariance_uses_the_pseudo_inverse() -> None:
    """A constant and a duplicated feature: the distance ignores their directions.

    The distance then equals the Mahalanobis distance of the two independent
    features alone.
    """
    X2, y = _correlated(150, seed=9, d=2)
    X_test2, _ = _correlated(50, seed=10, d=2)

    def extend(Z: np.ndarray) -> np.ndarray:
        return np.column_stack([Z, Z[:, 0], np.full(len(Z), 0.3)])

    model = CustomKNN(metric="mahalanobis").fit(extend(X2), y)
    assert model.rank_ == 2
    reduced = CustomKNN(metric="mahalanobis").fit(X2, y)
    dist, ind = model.kneighbors(extend(X_test2))
    ref_dist, ref_ind = reduced.kneighbors(X_test2)
    np.testing.assert_array_equal(ind, ref_ind)
    # Duplicating x_1 doubles its weight in Sigma, the pseudo-inverse undoes it.
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-8)
    np.testing.assert_allclose(
        model.predict_proba(extend(X_test2)), reduced.predict_proba(X_test2), atol=1e-8
    )


def test_rtol_drops_small_eigenvalues() -> None:
    """With a large rtol only the widest direction is kept."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 2)) * np.array([10.0, 1.0])
    y = rng.integers(0, 2, size=100)
    model = CustomKNN(metric="mahalanobis", rtol=0.5).fit(X, y)
    assert model.rank_ == 1
    np.testing.assert_allclose(
        model.precision_,
        np.linalg.pinv(model.covariance_, rtol=0.5, hermitian=True),
        rtol=1e-10,
        atol=1e-14,
    )
    # Only the widest direction v counts: d = |v.(x - x_i)| / sqrt(lambda).
    eigvals, eigvecs = np.linalg.eigh(model.covariance_)
    proj = X @ eigvecs[:, -1] / np.sqrt(eigvals[-1])
    dist, _ = model.kneighbors(X[:5])
    d_all = np.abs(proj[:5, None] - proj[None, :])
    np.testing.assert_allclose(
        dist, np.sort(d_all, axis=1)[:, : model.n_neighbors_], atol=1e-12
    )


def test_one_training_point() -> None:
    """Sigma = 0 has rank 0: every distance is 0 and the vote is its label."""
    model = CustomKNN(metric="mahalanobis").fit([[1.0, 2.0]], ["a"])
    assert model.rank_ == 0
    np.testing.assert_array_equal(model.predict([[5.0, -1.0]]), ["a"])
    np.testing.assert_array_equal(model.kneighbors([[5.0, -1.0]])[0], [[0.0]])


def test_euclidean_is_the_default() -> None:
    X, y = _correlated(100, seed=12)
    default = CustomKNN().fit(X, y)
    explicit = CustomKNN(metric="euclidean").fit(X, y)
    assert default.metric == "euclidean"
    assert not hasattr(default, "covariance_")
    np.testing.assert_array_equal(default.predict_proba(X), explicit.predict_proba(X))


def test_errors() -> None:
    with pytest.raises(ValueError, match="metric"):
        CustomKNN(metric="manhattan")
    with pytest.raises(ValueError, match="rtol"):
        CustomKNN(metric="mahalanobis", rtol=-1.0)
    model = CustomKNN(metric="mahalanobis").fit(np.eye(3), [0, 1, 1])
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.ones((2, 2)))
    with pytest.raises(ValueError, match="NaN"):
        model.predict([[np.nan, 0.0, 0.0]])


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_experiment_logs_both_knn(p: float) -> None:
    """exp_QELM_Q logs the two K-NN accuracies on <Z_k>, its split, no bias."""
    spec = importlib.util.spec_from_file_location("exp_QELM_Q", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = load_config(
        ["models=TFIM", "N=4", "samples.L=60", f"projective_measurement.p={p}"]
    )
    out = module.run(cfg)
    train = int(cfg.samples.train_size)
    Z_k, Y = out["X"][:, :-1], out["Y"]
    euclid = CustomKNN().fit(Z_k[:train], Y[:train])
    maha = CustomKNN(metric="mahalanobis").fit(Z_k[:train], Y[:train])
    assert out["bayes_estimate_knn_euclidean"] == euclid.score(Z_k[train:], Y[train:])
    assert out["bayes_estimate_knn_mahalanobis"] == maha.score(Z_k[train:], Y[train:])
    assert out["knn_covariance_rank"] == maha.rank_ == Z_k.shape[1]
    eig = np.linalg.eigvalsh(np.cov(Z_k[:train], rowvar=False))
    np.testing.assert_allclose(out["knn_covariance_cond"], eig[-1] / eig[0])
    for key in module.EXTRA_METRICS:
        assert key in out
