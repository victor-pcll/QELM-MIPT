"""Distribution of the linear observables fed to the QELM readout (issue #191).

The simple QELM (``experiments/QELM_Z/exp_QELM_Z.py``) evolves each sample
unitarily, without measurement, and reads the one-site expectation values
``<O_k> = <psi| O_k |psi>`` of the system qubits, for each Pauli ``O`` of
``experiment.observables``. Over the dataset, each ``<O_k>`` is a random
variable in ``[-1, 1]``: its distribution is the geometry of the features the
linear readout sees. This module separates these features, summarizes the
shape of their empirical distribution and compares it with two references:

- a Gaussian of the same mean and standard deviation;
- a Haar-random pure state, the limit of a fully scrambling unitary, drawn
  in the space the dynamics can reach (a symmetry sector of dimension
  ``D``). If the ``+1`` eigenspace of a one-site Pauli ``P`` has dimension
  ``k`` in that space, the weight of the state on it is
  ``B ~ Beta(k, D - k)`` and ``<P> = 2 B - 1``. On the full Hilbert space
  (``k = D / 2``) this has mean 0, variance ``1 / (D + 1)`` and excess
  kurtosis ``-6 / (D + 3)``.

The convergence of the empirical distribution with the number of samples
``n`` is measured by the two-sample Kolmogorov-Smirnov distance to the
largest dataset, which shrinks like ``1 / sqrt(n)`` (Glivenko-Cantelli).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def linear_observables(
    X: ArrayLike, labels: Sequence[str], Ns: int, *, bias: bool = True
) -> dict[str, np.ndarray]:
    """Split the feature matrix of ``exp_QELM_Z`` into one block per Pauli.

    ``exp_QELM_Z.run`` stores the features observable by observable:
    ``<O_1>, ..., <O_Ns>`` for the first label, then the next, then a bias
    column of ones.

    Args:
        X: Feature matrix of shape ``(L, len(labels) * Ns + bias)``.
        labels: Pauli labels of ``experiment.observables``, in order.
        Ns: Number of system qubits.
        bias: Whether the last column is the bias column, which is dropped.

    Returns:
        ``{label: values}``, ``values`` of shape ``(L, Ns)``: column ``k`` is
        ``<O>`` on system qubit ``k`` (chain site ``k + Nr``).

    Raises:
        ValueError: If ``X`` does not have that shape, or if the bias column
            is not all ones.
    """
    X = np.asarray(X, dtype=float)
    n_linear = len(labels) * Ns
    expected = n_linear + int(bias)
    if X.ndim != 2 or X.shape[1] != expected:
        raise ValueError(
            f"X must have shape (L, {expected}) for {len(labels)} observables on "
            f"Ns={Ns} qubits{' and a bias column' if bias else ''}, got {X.shape}."
        )
    if bias and not np.all(X[:, -1] == 1.0):
        raise ValueError("The last column of X is not the bias column of ones.")
    blocks = X[:, :n_linear].reshape(X.shape[0], len(labels), Ns)
    return {label: blocks[:, i] for i, label in enumerate(labels)}


def distribution_moments(values: ArrayLike) -> dict[str, float]:
    """Return the moments and the range of a sample of one feature.

    Args:
        values: Sample of shape ``(n,)``, ``n >= 2``.

    Returns:
        ``n``, ``mean``, ``sem`` (standard error of the mean), ``std``
        (with ``ddof = 1``), ``skewness``, ``excess_kurtosis`` (0 for a
        Gaussian, ``-1.2`` for a uniform law), ``min`` and ``max``. A constant
        sample has skewness and excess kurtosis ``nan``.

    Raises:
        ValueError: If ``values`` is not 1D or has fewer than 2 values.
    """
    v = _sample(values)
    std = float(np.std(v, ddof=1))
    constant = std == 0.0
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "sem": std / np.sqrt(v.size),
        "std": std,
        "skewness": np.nan if constant else float(stats.skew(v)),
        "excess_kurtosis": np.nan if constant else float(stats.kurtosis(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def haar_moments(dim: int, plus_dim: int | None = None) -> dict[str, float]:
    """Moments of ``<P>`` for a Haar-random pure state of dimension ``dim``.

    ``P`` is a Pauli whose ``+1`` eigenspace has dimension ``plus_dim`` in
    the space the state is drawn from. The weight of the state on it is
    ``B ~ Beta(plus_dim, dim - plus_dim)`` and ``<P> = 2 B - 1``.

    Args:
        dim: Dimension ``D`` of the space the state is drawn from, at least 2.
        plus_dim: Dimension of the ``+1`` eigenspace of ``P`` in it, in
            ``[1, dim - 1]``. ``None`` takes ``dim / 2`` (a one-site Pauli on
            the full Hilbert space, ``dim`` even).

    Returns:
        ``mean``, ``std``, ``skewness`` and ``excess_kurtosis``. With
        ``plus_dim = dim / 2``: 0, ``1 / sqrt(D + 1)``, 0 and ``-6 / (D + 3)``.
    """
    a, b = _beta_parameters(dim, plus_dim)
    mean, var, skew, kurt = (float(m) for m in stats.beta.stats(a, b, moments="mvsk"))
    return {
        "mean": 2 * mean - 1,
        "std": 2 * float(np.sqrt(var)),
        "skewness": skew,
        "excess_kurtosis": kurt,
    }


def haar_pdf(z: ArrayLike, dim: int, plus_dim: int | None = None) -> np.ndarray:
    """Density of ``<P>`` for a Haar-random pure state of dimension ``dim``.

    ``<P> = 2 B - 1`` with ``B ~ Beta(plus_dim, dim - plus_dim)``, so the
    density is ``Beta.pdf((z + 1) / 2) / 2``. With ``plus_dim = dim / 2`` it
    is proportional to ``(1 - z**2)**(D/2 - 1)``, uniform at ``D = 2`` (one
    qubit).

    Args:
        z: Points in ``[-1, 1]``, any shape.
        dim: Dimension ``D`` of the space the state is drawn from, at least 2.
        plus_dim: Dimension of the ``+1`` eigenspace of ``P`` in it;
            ``None`` takes ``dim / 2``.

    Returns:
        The density at ``z``, same shape, 0 outside ``[-1, 1]``.
    """
    a, b = _beta_parameters(dim, plus_dim)
    z = np.asarray(z, dtype=float)
    return np.asarray(stats.beta.pdf((z + 1) / 2, a, b) / 2)


def ks_to_normal(values: ArrayLike) -> float:
    """Kolmogorov-Smirnov distance to the Gaussian of the same mean and std.

    The parameters are estimated from the sample, so this is a distance, not
    a calibrated test (the p-value of the plain KS test would be too large).

    Args:
        values: Sample of shape ``(n,)``, ``n >= 2``.

    Returns:
        ``sup_z |F_n(z) - Phi((z - mean) / std)|`` in ``[0, 1]``; ``nan`` for a
        constant sample.
    """
    v = _sample(values)
    std = float(np.std(v, ddof=1))
    if std == 0.0:
        return float("nan")
    normal = stats.norm(loc=float(np.mean(v)), scale=std)
    return float(stats.kstest(v, normal.cdf).statistic)


def ks_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Two-sample Kolmogorov-Smirnov distance ``sup_z |F_a(z) - F_b(z)|``.

    Args:
        a: First sample, shape ``(n,)``.
        b: Second sample, shape ``(m,)``.

    Returns:
        The distance, in ``[0, 1]``.
    """
    return float(stats.ks_2samp(_sample(a), _sample(b)).statistic)


def convergence_table(
    samples: Mapping[int, ArrayLike], reference: ArrayLike | None = None
) -> list[dict[str, Any]]:
    """Moments and KS distance of each dataset size to a reference sample.

    Args:
        samples: ``{n_samples: values}``, the same feature from datasets of
            several sizes.
        reference: Sample the distances are measured to. ``None`` takes the
            largest dataset, whose own distance is then 0.

    Returns:
        One row per size, by increasing ``n``: ``n_samples``, the entries of
        :func:`distribution_moments`, ``ks_to_reference`` and
        ``ks_bound``, ``sqrt((n + m) / (n m))`` with ``m`` the size of the
        reference: the scale of the KS distance between two independent
        samples of the same law (``1.36`` times it is the 5 % critical value).
    """
    if not samples:
        raise ValueError("samples is empty.")
    ref = _sample(samples[max(samples)] if reference is None else reference)
    rows = []
    for n in sorted(samples):
        values = _sample(samples[n])
        rows.append(
            {
                "n_samples": int(n),
                **distribution_moments(values),
                "ks_to_reference": ks_distance(values, ref),
                "ks_bound": float(
                    np.sqrt((values.size + ref.size) / (values.size * ref.size))
                ),
            }
        )
    return rows


def _sample(values: ArrayLike) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError(f"Expected a 1D sample of at least 2 values, got {v.shape}.")
    return v


def _beta_parameters(dim: int, plus_dim: int | None) -> tuple[int, int]:
    if plus_dim is None:
        if dim < 2 or dim % 2:
            raise ValueError(f"dim must be an even integer >= 2, got {dim}.")
        plus_dim = dim // 2
    if not 0 < plus_dim < dim:
        raise ValueError(f"plus_dim must be in [1, dim - 1], got {plus_dim}, {dim}.")
    return plus_dim, dim - plus_dim


__all__ = [
    "convergence_table",
    "distribution_moments",
    "haar_moments",
    "haar_pdf",
    "ks_distance",
    "ks_to_normal",
    "linear_observables",
]
