"""Statistics of the ancilla entropy across measurement trajectories.

In the charge-sharpening setup (Agrawal et al., PRX 14, 041012 (2024),
appendix on charge sharpening, issue #137) each trajectory ``m`` of
measurement outcomes, drawn with its Born probability ``p_m``, leaves the
ancilla ``R`` with an entanglement entropy ``S_R^(m)``. The observables of
their Fig. 7 are moments of that distribution:

- the mean ``S_R = sum_m p_m S_R^(m)``;
- the central moments ``mu_k = sum_m p_m (S_R^(m) - S_R)**k``, with the
  variance ``mu_2``;
- the Binder ratio ``B = 1 - mu_4 / (3 mu_2**2)``.

The paper prints ``B = (1 - mu_4) / (3 mu_2**2)``, which is not
dimensionless; the formula here is the standard Binder cumulant, assumed to
be the intended one. ``B`` is 0 for a Gaussian distribution and ``2/3`` for
two equal peaks; its curves for several sizes cross near the transition.

Trajectories sampled with the Born rule are already weighted by ``p_m``, so
a plain average over samples estimates every ``sum_m p_m (...)``. Pass
``weights`` for samples that are not Born-distributed.
"""

from __future__ import annotations

import numpy as np


def _weights(S: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Return normalized weights for the samples ``S``, uniform by default."""
    if weights is None:
        return np.full(S.shape, 1.0 / S.size)
    w = np.asarray(weights, dtype=float)
    if w.shape != S.shape or np.any(w < 0) or not np.sum(w) > 0:
        raise ValueError(
            "weights must be non-negative, with the shape of S and a positive sum."
        )
    return w / np.sum(w)


def central_moment(S: np.ndarray, k: int, weights: np.ndarray | None = None) -> float:
    """Return the ``k``-th central moment ``mu_k = sum_m p_m (S_m - <S>)**k``.

    Args:
        S: Samples of shape ``(n,)``, ``n >= 1``.
        k: Order of the moment, ``k >= 1``.
        weights: Optional probabilities ``p_m`` of the samples; ``None`` gives
            every sample the weight ``1 / n``.

    Returns:
        ``mu_k``; ``mu_1`` is 0 up to rounding and ``mu_2`` is the variance
        with ``ddof=0``.

    Raises:
        ValueError: If ``S`` is empty or not 1-D, ``k < 1``, or the weights
            are invalid.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 1 or S.size == 0:
        raise ValueError(f"S must be a non-empty 1-D array, got shape {S.shape}.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    w = _weights(S, weights)
    mean = float(np.sum(w * S))
    return float(np.sum(w * (S - mean) ** k))


def binder_ratio(
    S: np.ndarray, weights: np.ndarray | None = None, tol: float = 1e-14
) -> float:
    """Return the Binder ratio ``B = 1 - mu_4 / (3 mu_2**2)`` of the samples.

    ``B`` is a ratio of moments, so it is meaningless when the samples are
    all equal up to rounding (``S_R ~ 1e-16`` at ``gamma = 1``): it is then
    ``nan``. When the distribution is a spike with rare outliers (``S_R``
    almost always 0 at strong measurement), ``mu_4 >> mu_2**2`` and ``B``
    is large and negative: that is a property of the distribution, not an
    error.

    Args:
        S: Samples of shape ``(n,)``.
        weights: Optional probabilities of the samples, as in
            :func:`central_moment`.
        tol: ``mu_2 <= tol`` counts as zero variance.

    Returns:
        ``B``, or ``nan`` when ``mu_2 <= tol``.
    """
    mu2 = central_moment(S, 2, weights)
    if mu2 <= tol:
        return float("nan")
    return 1.0 - central_moment(S, 4, weights) / (3.0 * mu2**2)


def sharpening_statistics(S: np.ndarray) -> dict[str, float]:
    """Return the mean, variance and Binder ratio of Born-sampled entropies.

    Args:
        S: Ancilla entropies ``S_R^(m)``, one per trajectory, shape ``(n,)``.

    Returns:
        ``S_R_mean``, ``S_R_var`` (``mu_2``) and ``binder`` (``B``).
    """
    return {
        "S_R_mean": float(np.mean(S)),
        "S_R_var": central_moment(S, 2),
        "binder": binder_ratio(S),
    }


def bootstrap_errors(
    S: np.ndarray,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Return bootstrap standard errors of :func:`sharpening_statistics`.

    The ``n`` samples are redrawn with replacement ``n_boot`` times; the
    error of each statistic is the standard deviation of its values over the
    resamples (``ddof=1``). Resamples with an undefined Binder ratio are
    skipped for ``binder``.

    Args:
        S: Samples of shape ``(n,)``.
        n_boot: Number of resamples, at least 2.
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        ``S_R_mean_err``, ``S_R_var_err`` and ``binder_err``.

    Raises:
        ValueError: If ``n_boot < 2``.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}.")
    S = np.asarray(S, dtype=float)
    gen = rng if rng is not None else np.random.default_rng(np.random.randint(2**31))
    stats: dict[str, list[float]] = {"S_R_mean": [], "S_R_var": [], "binder": []}
    for _ in range(n_boot):
        values = sharpening_statistics(S[gen.integers(0, S.size, size=S.size)])
        for key, value in values.items():
            stats[key].append(value)
    out: dict[str, float] = {}
    for key, samples in stats.items():
        arr = np.asarray(samples)
        arr = arr[np.isfinite(arr)]
        out[f"{key}_err"] = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return out
