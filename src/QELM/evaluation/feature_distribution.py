"""Distribution of the linear observables fed to the QELM readout (issue #191).

The unitary QELM (``experiments/QELM_Z/exp_QELM_Z.py``, or
``experiments/QELM_Q/exp_QELM_Q.py`` at ``p = 0``) evolves each sample
without measurement and reads the one-site expectation values
``<O_k> = Tr(O_k rho)`` of the system qubits, for each Pauli ``O``. Over
the dataset, each ``<O_k>`` is a random variable in ``[-1, 1]``: its
distribution is the geometry of the features the linear readout sees. This
module separates these features, summarizes the shape of their empirical
distribution and compares it with a Gaussian of the same mean and standard
deviation.

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


__all__ = [
    "convergence_table",
    "distribution_moments",
    "ks_distance",
    "ks_to_normal",
    "linear_observables",
]
