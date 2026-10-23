"""Double descent of a linear readout on fixed features (issue #181).

A readout ``W = Phi^+ y`` (:class:`~QELM.training.PseudoInverseReadout`,
no penalty) is fitted on ``n_train`` samples of a feature matrix with
``n_features = rank(Phi)`` independent columns. With
``alpha = n_train / n_features``:

- ``alpha < 1``: more weights than equations; the readout interpolates the
  training set (train error 0) with the smallest-norm weights;
- ``alpha = 1``: the square system is barely invertible, the smallest
  singular value of ``Phi_train`` is close to 0, ``||W||`` and the test
  error peak (the interpolation threshold);
- ``alpha > 1``: least squares, the test error decreases again towards its
  asymptote as ``n_train`` grows.

The peak sits at ``n_train = rank(Phi)``, not at the number of columns: two
linearly dependent observables (for instance ``<Y_i> = 0`` for a real
state) shift it. This module computes the features (quantum, closed form at
``t = 0``, and the classical random-feature baseline) and runs the sweep
over ``n_train``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from QELM.physics.operators import get_site_operator, pauli
from QELM.training.regression.pinv import PseudoInverseReadout

# Statistics of a sweep, each of shape (n_ratios, n_repeats).
SWEEP_STATISTICS = (
    "train_mse",
    "test_mse",
    "train_error",
    "test_error",
    "weight_norm",
)


def observable_operators(
    N: int,
    single: Sequence[str] = ("X", "Y", "Z"),
    pairs: Sequence[Sequence[str]] = (),
) -> tuple[list[str], list[sp.csr_matrix]]:
    """Return the measured observables as sparse matrices, with their names.

    The order is that of :class:`~QELM.training.SingleObservable` then
    :class:`~QELM.training.DoubleObservable`: for each label, sites
    ``0 .. N-1``; then for each pair ``(A, B)``, ``A_i B_j`` for ``i < j``
    in lexicographic order. Sparse, because a Pauli string has one non-zero
    entry per row: ``2**N`` entries instead of ``4**N``.

    Args:
        N: Number of qubits.
        single: One-site Pauli labels.
        pairs: Two-site label pairs.

    Returns:
        ``(names, operators)``: names such as ``"Z_3"`` or ``"XX_0_2"``, and
        the CSR matrices of shape ``(2**N, 2**N)``.
    """
    names: list[str] = []
    ops: list[sp.csr_matrix] = []
    for label in single:
        for i in range(N):
            names.append(f"{label}_{i}")
            ops.append(get_site_operator(pauli(label), i, N, sparse=True))
    for a, b in pairs:
        for i in range(N):
            A = get_site_operator(pauli(a), i, N, sparse=True)
            for j in range(i + 1, N):
                B = get_site_operator(pauli(b), j, N, sparse=True)
                names.append(f"{a}{b}_{i}_{j}")
                ops.append(sp.csr_matrix(A @ B))
    return names, ops


def expectations(psi: np.ndarray, ops: Sequence[sp.spmatrix]) -> np.ndarray:
    """Return ``<psi_n| O_k |psi_n>`` for every state and observable.

    Args:
        psi: States of shape ``(n_samples, dim)``, one per row.
        ops: Hermitian operators of shape ``(dim, dim)``.

    Returns:
        Real array of shape ``(n_samples, len(ops))``.
    """
    psi = np.asarray(psi, dtype=complex)
    out = np.empty((psi.shape[0], len(ops)))
    for k, op in enumerate(ops):
        # (O psi_n) for every row: psi @ O^T.
        o_psi = np.asarray(op @ psi.T).T
        out[:, k] = np.real(np.sum(psi.conj() * o_psi, axis=1))
    return out


def product_state_features(
    bloch: np.ndarray,
    single: Sequence[str] = ("X", "Y", "Z"),
    pairs: Sequence[Sequence[str]] = (),
) -> np.ndarray:
    """Return the observables of a product state from its Bloch vectors.

    For a product state ``<A_i> = r_i^A`` and ``<A_i B_j> = r_i^A r_j^B``
    (``i != j``). This is what the QELM measures at ``t = 0``, where the
    reservoir has done nothing: the features are fixed functions of the
    encoded input, in the order of :func:`observable_operators`.

    Args:
        bloch: Bloch vectors ``(<X_i>, <Y_i>, <Z_i>)`` of shape
            ``(n_samples, N, 3)``.
        single: One-site Pauli labels.
        pairs: Two-site label pairs.

    Returns:
        Real array of shape ``(n_samples, n_observables)``.
    """
    bloch = np.asarray(bloch, dtype=np.float64)
    axis = {"X": 0, "Y": 1, "Z": 2}
    N = bloch.shape[1]
    cols = [bloch[:, i, axis[label]] for label in single for i in range(N)]
    for a, b in pairs:
        cols += [
            bloch[:, i, axis[a]] * bloch[:, j, axis[b]]
            for i in range(N)
            for j in range(i + 1, N)
        ]
    return np.stack(cols, axis=1)


def add_bias(features: np.ndarray) -> np.ndarray:
    """Append a column of ones: the readout fits no intercept."""
    features = np.asarray(features, dtype=np.float64)
    return np.hstack([features, np.ones((features.shape[0], 1))])


def random_features(
    x: np.ndarray,
    n_hidden: int,
    rng: np.random.Generator,
    activation: str = "relu",
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Classical extreme learning machine: one fixed random hidden layer.

    ``h = g(x W + b)`` with ``W`` of shape ``(n_inputs, n_hidden)`` drawn
    from ``N(0, 1 / n_inputs)``, both ``W`` and ``b`` fixed (the "random
    kitchen sinks" of Rahimi and Recht); only the linear readout on top is
    trained. It is the classical baseline of the QELM, with as many hidden
    neurons as measured observables.

    With ``anchors``, neuron ``j`` gets ``b_j = -a_j W_j`` for a point
    ``a_j`` drawn from the anchors, so that its hyperplane ``x W_j + b_j = 0``
    crosses the data. Otherwise ``b`` is drawn from ``N(0, 1)``, and for a
    ReLU many neurons are then either never active (a zero column) or
    always active (an affine function of ``x``, so in the span of the
    inputs): the rank of the features falls far below ``n_hidden``.

    Args:
        x: Inputs of shape ``(n_samples, n_inputs)``.
        n_hidden: Number of hidden neurons.
        rng: Generator of the random weights.
        activation: ``"relu"``, ``"tanh"`` or ``"cos"``.
        anchors: Points of shape ``(n_anchors, n_inputs)`` the hyperplanes
            go through, typically the training inputs; ``None`` for random
            biases.

    Returns:
        Hidden activations of shape ``(n_samples, n_hidden)``.

    Raises:
        ValueError: If ``activation`` is unknown.
    """
    x = np.asarray(x, dtype=np.float64)
    W = rng.normal(scale=1 / np.sqrt(x.shape[1]), size=(x.shape[1], n_hidden))
    if anchors is None:
        b = rng.normal(size=n_hidden)
    else:
        anchors = np.asarray(anchors, dtype=np.float64)
        points = anchors[rng.integers(len(anchors), size=n_hidden)]
        b = -np.einsum("ji,ij->j", points, W)
    z = x @ W + b
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "tanh":
        return np.tanh(z)
    if activation == "cos":
        return np.cos(z)
    raise ValueError(f"activation must be relu, tanh or cos, got {activation!r}.")


def numerical_rank(features: np.ndarray, rtol: float | None = None) -> int:
    """Return the number of linearly independent columns of ``features``.

    Args:
        features: Matrix of shape ``(n_samples, n_features)``, with more
            samples than columns for the rank to be that of the observables.
        rtol: Cutoff on the singular values relative to the largest; ``None``
            for NumPy's default.

    Returns:
        The numerical rank.
    """
    return int(np.linalg.matrix_rank(features, rtol=rtol))


def n_train_grid(alpha_ratios: Sequence[float], n_features: int) -> np.ndarray:
    """Return ``n_train = round(alpha * n_features)``, at least 1, for each ratio.

    Args:
        alpha_ratios: Ratios ``n_train / n_features``.
        n_features: Number of independent features, the rank.

    Returns:
        Integer array of the same length as ``alpha_ratios``. A ratio of
        exactly 1 gives exactly ``n_features``.
    """
    ratios = np.asarray(alpha_ratios, dtype=np.float64)
    return np.maximum(1, np.rint(ratios * n_features)).astype(np.int64)


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of ``sign(y_pred)`` different from the labels ``+-1``.

    A prediction of exactly 0 counts as ``+1``.
    """
    labels = np.where(np.asarray(y_pred) >= 0, 1.0, -1.0)
    return float(np.mean(labels != np.asarray(y_true)))


@dataclass
class SweepResult:
    """Train and test scores of the readout against ``n_train``.

    Attributes:
        n_features: Rank of the feature matrix of the pool.
        n_columns: Number of columns, bias included.
        n_train: Training-set sizes, shape ``(n_ratios,)``.
        alpha: ``n_train / n_features``, shape ``(n_ratios,)``.
        train_mse, test_mse: Mean squared error of the readout output
            against the labels ``+-1``, shape ``(n_ratios, n_repeats)``.
        train_error, test_error: Classification error of ``sign(output)``.
        weight_norm: ``||W||_2`` of the fitted readout.
    """

    n_features: int
    n_columns: int
    n_train: np.ndarray
    alpha: np.ndarray
    train_mse: np.ndarray
    test_mse: np.ndarray
    train_error: np.ndarray
    test_error: np.ndarray
    weight_norm: np.ndarray

    def peak_alpha(self) -> float:
        """Return the ratio where the median test MSE over the repeats is largest."""
        return float(self.alpha[int(np.argmax(np.median(self.test_mse, axis=1)))])


def double_descent_sweep(
    features_pool: np.ndarray,
    y_pool: np.ndarray,
    features_test: np.ndarray,
    y_test: np.ndarray,
    alpha_ratios: Sequence[float],
    n_repeats: int,
    rng: np.random.Generator,
    rcond: float | None = None,
    n_features: int | None = None,
) -> SweepResult:
    """Fit the pseudo-inverse readout on random subsets of the pool.

    For each ratio, ``n_repeats`` training sets of ``n_train`` samples are
    drawn from the pool without replacement; the readout is fitted on each
    and scored on it and on the whole test set.

    Args:
        features_pool: Features of the training pool, ``(n_pool, n_columns)``.
        y_pool: Labels ``+-1`` of the pool.
        features_test: Features of the test set, ``(n_test, n_columns)``.
        y_test: Labels of the test set.
        alpha_ratios: Ratios ``n_train / n_features`` to scan.
        n_repeats: Training sets drawn per ratio.
        rng: Generator of the training sets.
        rcond: Cutoff of the pseudo-inverse (``None`` for NumPy's default).
        n_features: Number of independent features; ``None`` computes the
            rank of ``features_pool``.

    Returns:
        The :class:`SweepResult`.

    Raises:
        ValueError: If the largest ``n_train`` exceeds the pool.
    """
    features_pool = np.asarray(features_pool, dtype=np.float64)
    features_test = np.asarray(features_test, dtype=np.float64)
    if n_features is None:
        n_features = numerical_rank(features_pool)
    n_train = n_train_grid(alpha_ratios, n_features)
    n_pool = features_pool.shape[0]
    if n_train.max() > n_pool:
        raise ValueError(
            f"The largest training set ({n_train.max()} samples, alpha = "
            f"{max(alpha_ratios)} x {n_features} features) exceeds the pool of "
            f"{n_pool} samples: enlarge the pool or lower the largest ratio."
        )
    stats = {key: np.empty((n_train.size, n_repeats)) for key in SWEEP_STATISTICS}
    readout = PseudoInverseReadout({"rcond": rcond})
    for a, n in enumerate(n_train):
        for r in range(n_repeats):
            subset = rng.choice(n_pool, size=int(n), replace=False)
            phi, y = features_pool[subset], y_pool[subset]
            readout.fit(phi, y)
            assert readout.W is not None
            fit, pred = readout.predict(phi), readout.predict(features_test)
            stats["train_mse"][a, r] = float(np.mean((fit - y) ** 2))
            stats["test_mse"][a, r] = float(np.mean((pred - y_test) ** 2))
            stats["train_error"][a, r] = classification_error(y, fit)
            stats["test_error"][a, r] = classification_error(y_test, pred)
            stats["weight_norm"][a, r] = float(np.linalg.norm(readout.W))
    return SweepResult(
        n_features=n_features,
        n_columns=features_pool.shape[1],
        n_train=n_train,
        alpha=n_train / n_features,
        **stats,
    )


def full_pool_accuracy(
    features_pool: np.ndarray,
    y_pool: np.ndarray,
    features_test: np.ndarray,
    y_test: np.ndarray,
    rcond: float | None = None,
) -> float:
    """Test accuracy of the pseudo-inverse readout fitted on the whole pool."""
    readout = PseudoInverseReadout({"rcond": rcond})
    readout.fit(features_pool, y_pool)
    return 1.0 - classification_error(y_test, readout.predict(features_test))
