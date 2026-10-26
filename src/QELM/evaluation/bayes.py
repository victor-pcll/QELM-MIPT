"""Bayes posterior of a two-class label from continuous features (issue #194).

The readout of the QELM is a linear regression: it says how much of the
label a hyperplane of the features carries, not how much the features carry
at all. The best any classifier can do is the Bayes classifier, which picks
the label of largest posterior ``P(y | x)``. That posterior needs the
class-conditional densities ``f_k(x) = P(x | y = k)``, which are not known
for the ``<Z_k>`` of a quantum reservoir. Quadratic discriminant analysis
(QDA) estimates them as Gaussians, one mean and one covariance per class,
from the training samples, and scores the test samples with::

    P(y = k | x) = pi_k f_k(x) / sum_k' pi_k' f_k'(x),

``pi_k`` the share of class ``k`` in the training set. This is the Bayes
classifier exactly when the ``f_k`` are Gaussian, and its estimate of the
ceiling otherwise: a non-Gaussian density can hold more than QDA finds.

:func:`qda_posterior` is the estimator (scikit-learn's
``QuadraticDiscriminantAnalysis``); :func:`correct_probability` reads
``P_corr = P(y* | x)``, the posterior of the true label ``y*``, as in
:mod:`QELM.evaluation.learnability`; :func:`posterior_scores` turns it into
the accuracy, the balanced accuracy and the mean binary entropy
``E[H(P_corr)]``.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.discriminant_analysis import (  # type: ignore[import-untyped]
    QuadraticDiscriminantAnalysis,
)

from QELM.evaluation.learnability import binary_entropy


def _features(X: np.ndarray, name: str) -> np.ndarray:
    """Return ``X`` as a 2D float array, a 1D input becoming one column."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be (n, d) and not empty, got {arr.shape}.")
    return arr


def qda_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    reg_param: float = 1e-6,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a QDA on the training samples and return the test posteriors.

    Args:
        X_train: Training features, ``(n_train, d)``, or ``(n_train,)`` for a
            single feature.
        y_train: Training labels, ``(n_train,)``, at least two classes, each
            with at least two samples.
        X_test: Test features, ``(n_test, d)`` or ``(n_test,)``.
        reg_param: Regularization of the covariances, added to their
            eigenvalues (``reg_param`` of scikit-learn). A small value keeps
            the inverse finite when the features of a class lie on a
            hyperplane, e.g. ``sum_k <Z_k>`` fixed by a conserved charge.
        tol: Smallest eigenvalue of a regularized covariance that counts as
            nonzero; the fit fails below it. scikit-learn's default, 1e-4,
            is larger than the variances of the ``<Z_k>`` at large ``h``,
            where the evolution barely moves them, so it is lowered here.

    Returns:
        ``classes`` (sorted labels, ``(K,)``) and ``proba``, the posteriors
        ``P(y = classes[k] | x)`` of the test samples, ``(n_test, K)``, each
        row summing to 1.

    Raises:
        ValueError: If the shapes do not match, there are fewer than two
            classes, or ``reg_param`` is not in ``[0, 1]``.
    """
    Xtr = _features(X_train, "X_train")
    Xte = _features(X_test, "X_test")
    y = np.asarray(y_train)
    if y.shape != (Xtr.shape[0],):
        raise ValueError(f"y_train must be ({Xtr.shape[0]},), got {y.shape}.")
    if Xte.shape[1] != Xtr.shape[1]:
        raise ValueError(
            f"X_test has {Xte.shape[1]} features, X_train has {Xtr.shape[1]}."
        )
    if np.unique(y).size < 2:
        raise ValueError("y_train must hold at least two classes.")
    if not 0.0 <= reg_param <= 1.0:
        raise ValueError(f"reg_param must be in [0, 1], got {reg_param}.")
    qda = QuadraticDiscriminantAnalysis(reg_param=reg_param, tol=tol)
    qda.fit(Xtr, y)
    proba = qda.predict_proba(Xte)
    return np.asarray(qda.classes_), np.asarray(proba, dtype=float)


def correct_probability(
    classes: np.ndarray, proba: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """Return ``P_corr = P(y* | x)``, the posterior of each true label.

    Args:
        classes: Labels of the columns of ``proba``, ``(K,)``.
        proba: Posteriors, ``(n, K)``.
        y_true: True labels, ``(n,)``, each one of ``classes``.

    Returns:
        Array of shape ``(n,)`` with values in ``[0, 1]``.

    Raises:
        ValueError: If a label of ``y_true`` is not in ``classes``.
    """
    classes = np.asarray(classes)
    y = np.asarray(y_true)
    column = np.searchsorted(classes, y)
    column = np.clip(column, 0, classes.size - 1)
    if not np.all(classes[column] == y):
        raise ValueError("Every label of y_true must be one of classes.")
    return np.asarray(proba[np.arange(y.size), column], dtype=float)


def posterior_scores(
    classes: np.ndarray,
    proba: np.ndarray,
    y_true: np.ndarray,
    base: float = 2.0,
) -> dict[str, float]:
    """Score the posteriors of a two-class problem.

    Args:
        classes: Labels of the columns of ``proba``, ``(2,)``.
        proba: Posteriors, ``(n, 2)``.
        y_true: True labels, ``(n,)``.
        base: Logarithm base of the entropy, 2 (default) for bits.

    Returns:
        - ``accuracy``: share of samples whose label of largest posterior is
          the true one;
        - ``balanced_accuracy``: mean over the true classes of that share;
        - ``P_corr_mean``: mean posterior of the true label;
        - ``entropy_mean``: mean binary entropy ``E[H(P_corr)]``, 1 bit when
          the posterior is 1/2 everywhere (nothing learned), 0 when it is
          certain.

    Raises:
        ValueError: If there are not exactly two classes.
    """
    classes = np.asarray(classes)
    if classes.size != 2 or proba.shape[1] != 2:
        raise ValueError(f"Expected two classes, got {classes.size}.")
    y = np.asarray(y_true)
    pred = classes[np.argmax(proba, axis=1)]
    right = pred == y
    P = correct_probability(classes, proba, y)
    return {
        "accuracy": float(np.mean(right)),
        "balanced_accuracy": float(np.mean([np.mean(right[y == c]) for c in classes])),
        "P_corr_mean": float(np.mean(P)),
        "entropy_mean": float(np.mean(binary_entropy(P, base))),
    }


def prior_entropy(y_train: np.ndarray, base: float = 2.0) -> float:
    """Return the entropy of the class shares of ``y_train``, two classes.

    The mean entropy of a classifier that ignores the features and answers
    the training shares ``pi_k``: the chance level of ``entropy_mean`` in
    :func:`posterior_scores`, 1 bit for balanced classes.

    Args:
        y_train: Training labels, ``(n,)``, at most two classes.
        base: Logarithm base, 2 (default) for bits.

    Returns:
        ``H(pi) = -sum_k pi_k log pi_k`` in base ``base``.

    Raises:
        ValueError: If ``y_train`` is empty or has more than two classes.
    """
    _, counts = np.unique(np.asarray(y_train), return_counts=True)
    if counts.size == 0 or counts.size > 2:
        raise ValueError(f"Expected one or two classes, got {counts.size}.")
    pi = counts / counts.sum()
    return float(-np.sum(pi * np.log(pi)) / math.log(base))
