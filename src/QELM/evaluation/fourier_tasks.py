"""QELM scores on targets of known Fourier degree (issue #309).

:mod:`QELM.evaluation.fourier` says which functions of the input the mean
features can represent. This module asks the QELM to learn such functions
and compares its scores with that prediction:

- :func:`degree_target`: a target ``y(theta) = sum_alpha yhat_alpha
  B_alpha(theta)`` spread over the Bloch monomials of one degree ``k``,
  centered and of unit variance for angles uniform on ``[low, high]``;
- :func:`ideal_r2`: the R2 of the best linear readout of the **mean**
  features, for infinitely many samples and no shot noise, from ``C(p)``
  alone: the share of the target in reach;
- :func:`sample_records`: single-shot records ``M_{t,i}`` and final Z
  outcomes, drawn with the Born rule on the state vector, exactly the
  features whose means are the rows of ``C(p)``;
- :func:`qelm_scores`: the ridge readout of these records, scored by the
  test R2 and the test accuracy of the sign of ``y``.

The gap between :func:`ideal_r2` and the test R2 is the cost of the shot
noise of a single record (and of the finite training set).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from QELM.evaluation.fourier import (
    _contract_sites,
    angle_moments,
    feature_covariance,
    monomial_degrees,
)
from QELM.evaluation.metrics import R2, accuracy
from QELM.models.circuits.base import Layer, apply_layer
from QELM.models.circuits.monitored import measure_z
from QELM.training.regression.ridge import RidgeReadout


def _mean_vector(N: int, low: float, high: float) -> np.ndarray:
    """Return ``E[B_alpha]`` for i.i.d. angles uniform on ``[low, high]``."""
    m, _ = angle_moments(low, high)
    out = np.ones(1)
    for _ in range(N):
        out = np.kron(out, m)
    return out


def _gram_apply(v: np.ndarray, N: int, low: float, high: float) -> np.ndarray:
    """Return ``G v`` with ``G = E[B B^T]``, contracted site by site."""
    _, G = angle_moments(low, high)
    return _contract_sites(v[None, :], N, G)[0]


def target_variance(
    yhat: np.ndarray, N: int, low: float = 0.0, high: float = np.pi
) -> float:
    """Return ``Var(y)`` for ``y = B . yhat`` and uniform angles."""
    mean = float(_mean_vector(N, low, high) @ yhat)
    return float(yhat @ _gram_apply(yhat, N, low, high)) - mean**2


def degree_target(
    N: int,
    k: int,
    rng: np.random.Generator,
    low: float = 0.0,
    high: float = np.pi,
) -> tuple[np.ndarray, float]:
    """Draw a target of degree ``k``: Gaussian weights on the degree-``k`` monomials.

    Returns:
        ``(yhat, offset)``: the target is ``y(theta) = B(theta) . yhat -
        offset``, of mean 0 and variance 1 for uniform angles.
    """
    if not 1 <= k <= N:
        raise ValueError(f"k must lie in [1, N], got k={k}, N={N}.")
    yhat = np.zeros(3**N)
    mask = monomial_degrees(N) == k
    yhat[mask] = rng.normal(size=int(mask.sum()))
    yhat /= np.sqrt(target_variance(yhat, N, low, high))
    offset = float(_mean_vector(N, low, high) @ yhat)
    return yhat, offset


def ideal_r2(
    C: np.ndarray,
    yhat: np.ndarray,
    N: int,
    low: float = 0.0,
    high: float = np.pi,
    rcond: float = 1e-10,
) -> float:
    """R2 of the best linear readout of the mean features ``C B(theta)``.

    ``R2 = c^T Sigma^+ c / Var(y)`` with ``Sigma`` the covariance of the
    mean features and ``c = Cov(features, y)``, for angles uniform on
    ``[low, high]``: no shot noise, infinitely many samples.
    """
    var = target_variance(yhat, N, low, high)
    if var <= 0:
        return 0.0
    mvec = _mean_vector(N, low, high)
    cov = C @ _gram_apply(yhat, N, low, high) - (C @ mvec) * float(mvec @ yhat)
    Sigma = feature_covariance(C, N, low, high)
    return float(cov @ np.linalg.pinv(Sigma, rcond=rcond, hermitian=True) @ cov / var)


def product_state(theta: np.ndarray) -> np.ndarray:
    """State vector of the angle encoding, ``cos(t/2)|0> + sin(t/2)|1>`` per qubit."""
    psi = np.ones(1, dtype=complex)
    for t in theta:
        psi = np.kron(psi, [np.cos(t / 2), np.sin(t / 2)])
    return psi


def sample_records(
    layers: Sequence[Layer],
    N: int,
    p: float,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one single-shot record per input, with the Born rule.

    After each layer, each site is measured in Z with probability ``p``;
    ``M_{t,i} = +1`` (outcome 0) or ``-1`` (outcome 1) if measured, ``0``
    otherwise. Then every site is measured once more: the final Z outcomes.

    Args:
        layers: The brickwork layers, as in
            :func:`QELM.evaluation.fourier.transfer_matrix`.
        N: Number of qubits.
        p: Measurement probability.
        theta: Angles, shape ``(n, N)``.
        rng: Random generator of the masks and outcomes.

    Returns:
        Shape ``(n, T N + N)``, columns ordered as the rows of ``C(p)``.
    """
    theta = np.atleast_2d(theta)
    T = len(layers)
    out = np.zeros((theta.shape[0], T * N + N))
    for s, angles in enumerate(theta):
        psi = product_state(angles)
        for t in range(T):
            psi = apply_layer(psi, layers[t])
            for site in np.flatnonzero(rng.random(N) < p):
                psi, outcome = measure_z(psi, int(site), rng)
                out[s, t * N + int(site)] = 1 - 2 * outcome
        for i in range(N):
            psi, outcome = measure_z(psi, i, rng)
            out[s, T * N + i] = 1 - 2 * outcome
    return out


def qelm_scores(
    records: np.ndarray, y: np.ndarray, n_train: int, alpha: float = 1e-3
) -> dict[str, float]:
    """Fit a ridge (with a bias column) on the first ``n_train`` samples.

    Returns:
        Test and train R2 and the test accuracy of the sign of ``y``.
    """
    phi = np.hstack([records, np.ones((records.shape[0], 1))])
    readout = RidgeReadout({"alpha": alpha})
    readout.fit(phi[:n_train], y[:n_train])
    pred = readout.predict(phi)
    return {
        "r2_test": R2(y[n_train:], pred[n_train:]),
        "r2_train": R2(y[:n_train], pred[:n_train]),
        "accuracy_test": accuracy(y[n_train:], pred[n_train:]),
    }


def shot_noise_covariance(
    records: np.ndarray, C: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Estimate ``E_theta[Cov(M | theta)]``, the shot noise of one record.

    Unbiased: the mean over the samples of ``r r^T - phi phi^T``, with ``r``
    the sampled record and ``phi = C B(theta)`` its exact mean.
    """
    from QELM.evaluation.fourier import bloch_monomials

    phi = bloch_monomials(np.atleast_2d(theta)) @ C.T
    n = records.shape[0]
    return np.asarray((records.T @ records - phi.T @ phi) / n)


def predicted_r2(
    C: np.ndarray,
    yhat: np.ndarray,
    noise: np.ndarray,
    shots: float,
    N: int,
    low: float = 0.0,
    high: float = np.pi,
    rcond: float = 1e-10,
) -> float:
    """R2 of the best linear readout of records averaged over ``shots`` shots.

    ``R2 = c^T (Sigma + noise / shots)^+ c / Var(y)``: the population
    optimum, for infinitely many inputs, of a readout of features whose
    means are ``C B(theta)`` and whose shot noise is ``noise / shots``.
    ``shots = inf`` gives :func:`ideal_r2`.
    """
    var = target_variance(yhat, N, low, high)
    if var <= 0:
        return 0.0
    mvec = _mean_vector(N, low, high)
    cov = C @ _gram_apply(yhat, N, low, high) - (C @ mvec) * float(mvec @ yhat)
    A = feature_covariance(C, N, low, high)
    if np.isfinite(shots):
        A = A + noise / shots
    return float(cov @ np.linalg.pinv(A, rcond=rcond, hermitian=True) @ cov / var)


def gaussian_accuracy(r2: float) -> float:
    """Sign accuracy of a readout with correlation ``sqrt(R2)`` to a Gaussian target.

    ``1/2 + arcsin(rho) / pi`` for a centered bivariate Gaussian (Sheppard).
    """
    rho = float(np.sqrt(np.clip(r2, 0.0, 1.0)))
    return 0.5 + float(np.arcsin(rho)) / np.pi


def product_features(
    records: np.ndarray, products: Sequence[Sequence[int]]
) -> np.ndarray:
    """Return the sampled product features, one column per product.

    ``records`` as returned by :func:`sample_records`; column ``r`` is the
    product of the columns ``products[r]``, a single-shot sample of the row
    ``r`` of :func:`QELM.evaluation.fourier.product_transfer_matrix`.
    """
    out = np.ones((records.shape[0], len(products)))
    for r, prod in enumerate(products):
        for k in prod:
            out[:, r] *= records[:, k]
    return out
