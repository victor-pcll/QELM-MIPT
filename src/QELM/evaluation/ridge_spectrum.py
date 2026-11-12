"""Spectrum of the effects and conditioning of the ridge readout (issue #334).

Two hypotheses relate the measurement-induced transition (MIPT) to the ridge
readout of the QELM: (A) the spectrum of the effects ``E_m`` sets the
conditioning of the ridge problem, and (B) the MIPT is a bias/variance
transition of the ridge. This module measures both on the same circuits, at
two levels.

**Inputs.** Angle encoding, ``rho(theta) = prod_i (I + sin(theta_i) X_i +
cos(theta_i) Z_i) / 2``, with angles uniform on the **full circle**
``[0, 2 pi)``. The Bloch monomials ``B_a`` of :mod:`QELM.evaluation.fourier`
are then orthogonal, ``E[B_a B_b] = w_a delta_ab`` with ``w_a = 2^{-|a|}``,
and the mean input is the maximally mixed state. The functions
``f_a = B_a / sqrt(w_a)`` (``a != 0``) are an orthonormal basis of the
centered targets, and a target ``y = sum_a beta_a B_a`` has the coordinates
``sqrt(w_a) beta_a``.

**Level I: the full record.** A record ``m`` (mid-circuit outcomes, with or
without the final ``Z`` readout of every site) has
``Pr(m | theta) = sum_a c_a(m) B_a(theta)``, ``c_a`` the Pauli coefficients
of its effect. Records are drawn with the Born rule of the mixed input, the
mean of the input ensemble, so ``Pr(m) = c_0(m)``. The conditional mean of
``f_a`` given the record is ``v_a(m) = sqrt(w_a) c_a(m) / c_0(m)``, and

    F = E_m[v(m) v(m)^T]

is the covariance of these conditional means. Its eigenvalues ``mu_k`` lie
in ``[0, 1]``: they are the squared canonical correlations between one shot
of the record and the input. With ``K`` shots of the same input, the best
linear readout of the histogram of records reaches, on the eigendirection
``k``,

    R2_k(K) = K mu_k / (1 + (K - 1) mu_k),

so a direction is learnable (``R2_k > 1/2``) when ``mu_k > 1 / (K + 1)``:
``n_learn(K)`` counts them. ``F`` replaces the covariance ``X^T X / S`` of
the hypothesis, whose raw condition number is infinite at every ``p``
(``sum_m Pr(m | theta) = 1`` and more records than the rank): what matters
is the signal spectrum against the shot-noise floor. The span of ``F`` is the
span of the effects, computed exactly by the backward recursion of
:mod:`QELM.evaluation.effect_rank`; the weight of a target outside it,
``E_perp``, is the strict approximation bias, which no number of shots or
samples removes.

**Level II: the local features of the QELM.** The features are the record
``M_{t,i}`` (``+-1`` if measured, ``0`` otherwise) and the final ``Z_i``, as
in :mod:`QELM.evaluation.fourier_tasks`, with a fresh mask for every shot,
averaged over ``K`` shots of the same input. A ridge readout is trained on
them and its test error is split into the squared bias and the variance over
independent shot draws, at each regularisation ``lambda``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QELM.evaluation.effect_mipt import pauli_coefficients, sample_kraus
from QELM.evaluation.effect_rank import (
    _orthonormal_rows,
    apply_gate,
    brick_pairs,
    final_readout_basis,
    heisenberg_transfer,
    measure_site,
    no_y_columns,
)
from QELM.evaluation.fourier import bloch_monomials, monomial_degrees
from QELM.evaluation.fourier_tasks import product_state
from QELM.models.circuits.base import Layer

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Conventions shared by the two levels
# ---------------------------------------------------------------------------


def circle_weights(n: int) -> np.ndarray:
    """Return ``w_a = E[B_a^2] = 2^{-|a|}`` for angles uniform on the circle."""
    return np.asarray(0.5 ** monomial_degrees(n), dtype=float)


def layers_from_gates(gates: Sequence[Sequence[np.ndarray]], n: int) -> list[Layer]:
    """Layers ``[(i, i + 1, U), ...]`` of the gates of :func:`draw_haar_circuit`.

    The order of the gates in a layer is that of
    :func:`QELM.evaluation.effect_rank.brick_pairs`, the order used by
    :func:`QELM.evaluation.effect_mipt.sample_kraus`, so that both levels
    run the same circuit.
    """
    pairs = brick_pairs(n)
    return [
        [(i, i + 1, np.asarray(g)) for i, g in zip(pairs, layer, strict=True)]
        for layer in gates
    ]


def single_site_target(n: int, site: int, axis: str) -> np.ndarray:
    """Coefficients of ``y = sqrt(2) sin(theta_site)`` (``"x"``) or ``cos`` (``"z"``).

    Unit variance and zero mean for angles uniform on the circle.
    """
    if axis not in ("x", "z"):
        raise ValueError(f"axis must be 'x' or 'z', got {axis!r}.")
    digit = 1 if axis == "x" else 2
    index = digit * 3 ** (n - 1 - site)
    out = np.zeros(3**n)
    out[index] = np.sqrt(2.0)
    return out


def degree_target_circle(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian weights on the degree-``k`` monomials, unit variance on the circle."""
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, n], got k={k}, n={n}.")
    beta = np.zeros(3**n)
    mask = monomial_degrees(n) == k
    beta[mask] = rng.normal(size=int(mask.sum()))
    var = float(np.sum(circle_weights(n) * beta**2))
    return beta / np.sqrt(var)


def target_values(beta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Values ``y(theta) = B(theta) . beta``."""
    return np.asarray(bloch_monomials(np.atleast_2d(theta)) @ beta)


# ---------------------------------------------------------------------------
# Level I: spectrum of the effects of the full record
# ---------------------------------------------------------------------------


def sample_effect(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    final: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Effect of one Born-drawn record of the mixed input, trace 1.

    Args:
        gates: ``gates[t][g]``, as for :func:`sample_kraus`.
        masks: Boolean ``(T, n)`` mask of the mid-circuit measurements.
        final: Add the final ``Z`` readout of every site to the record; the
            effect is then of rank one.
        rng: Random generator of the outcomes.

    Returns:
        ``E_m / Tr E_m``, a ``(2^n, 2^n)`` matrix.
    """
    kraus = sample_kraus(gates, masks, rng)
    if not final:
        effect = kraus.conj().T @ kraus
        return np.asarray(effect / np.trace(effect).real)
    # Given the mid-circuit record, Pr(z) = <z| K K^dagger |z> for the mixed
    # input (Tr K^dagger K = 1); the effect of (m, z) is K^dagger |z><z| K.
    probs = np.sum(np.abs(kraus) ** 2, axis=1)
    z = int(rng.choice(probs.size, p=probs / probs.sum()))
    phi = kraus[z].conj()
    effect = np.outer(phi, phi.conj())
    return np.asarray(effect / np.trace(effect).real)


def conditional_means(effect: np.ndarray) -> np.ndarray:
    """Return ``v_a = sqrt(w_a) c_a / c_0`` over the ``3^n`` monomials (``v_0 = 1``).

    ``c_a`` are the Pauli coefficients of the effect on the strings without
    ``Y``, in the monomial order of :mod:`QELM.evaluation.fourier`.
    """
    n = effect.shape[0].bit_length() - 1
    c = pauli_coefficients(effect)[no_y_columns(n)]
    return np.asarray(np.sqrt(circle_weights(n)) * c / c[0])


@dataclass(frozen=True)
class RecordSpectrum:
    """Spectrum of ``F`` estimated from Born-drawn records.

    Attributes:
        mu: Eigenvalues of ``F``, decreasing (at most the number of records).
        vectors: Matching eigenvectors as columns, in the coordinates
            ``f_a`` (``a != 0``).
        n_records: Number of records of the estimate.
    """

    mu: np.ndarray
    vectors: np.ndarray
    n_records: int


def record_spectrum(v: np.ndarray) -> RecordSpectrum:
    """Spectrum of ``F = E[v v^T]`` from rows ``v`` of :func:`conditional_means`.

    The constant column is dropped. The estimate is the sample mean, which
    is unbiased for ``F``; its smallest eigenvalues are biased down when the
    number of records is not large against ``3^n``.
    """
    x = np.asarray(v, dtype=float)[:, 1:] / np.sqrt(v.shape[0])
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    return RecordSpectrum(mu=s**2, vectors=vt.T, n_records=int(v.shape[0]))


def n_learn(mu: np.ndarray, shots: float) -> int:
    """Number of eigendirections with ``R2_k(K) > 1/2``: ``mu_k > 1 / (K + 1)``."""
    if np.isinf(shots):
        return int(np.sum(mu > 1e-12))
    return int(np.sum(mu > 1.0 / (shots + 1.0)))


def spectral_r2(spectrum: RecordSpectrum, beta: np.ndarray, shots: float) -> float:
    """R2 of the best readout of the ``K``-shot histogram, population limit.

    ``sum_k z_k^2 K mu_k / (1 + (K - 1) mu_k)``, with ``z`` the coordinates of
    the target on the eigenvectors of ``F``; ``K = inf`` gives the weight of
    the target on the estimated span.
    """
    n = int(round(np.log(beta.size) / np.log(3)))
    coords = (np.sqrt(circle_weights(n)) * beta)[1:]
    z2 = (spectrum.vectors.T @ coords) ** 2
    mu = spectrum.mu
    if np.isinf(shots):
        gain = (mu > 1e-12).astype(float)
    else:
        gain = shots * mu / (1.0 + (shots - 1.0) * mu)
    return float(np.sum(z2 * gain))


def single_shot_r2(v: np.ndarray, beta: np.ndarray) -> float:
    """Unbiased estimate of ``R2(K = 1) = E_m[E[y | m]^2]`` for a unit target."""
    n = int(round(np.log(beta.size) / np.log(3)))
    coords = np.sqrt(circle_weights(n)) * beta
    coords[0] = 0.0
    cond = np.asarray(v) @ coords
    return float(np.mean(cond**2))


def effect_span(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    final: bool,
    tol: float = 1e-6,
) -> np.ndarray:
    """Orthonormal basis of the span of the effects seen by the angle encoding.

    Backward recursion of :func:`QELM.evaluation.effect_rank.effect_rank`,
    started from the diagonal strings (``final``) or the identity alone
    (mid-circuit record only), then projected on the strings without ``Y``
    and orthonormalised in the coordinates ``f_a`` (weights ``sqrt(w_a)``).

    Returns:
        Rows of shape ``(rank, 3^n)`` in the coordinates ``f_a`` (the
        constant included).
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    if final:
        basis = final_readout_basis(n)
    else:
        basis = np.zeros((1, 4**n))
        basis[0, 0] = 1.0
    pairs = brick_pairs(n)
    for t in reversed(range(depth)):
        for measured in np.flatnonzero(masks[t]):
            basis = measure_site(basis, n, int(measured), tol)
        for left, gate in zip(reversed(pairs), reversed(gates[t]), strict=True):
            basis = apply_gate(basis, n, left, heisenberg_transfer(gate))
    angle = basis[:, no_y_columns(n)] * np.sqrt(circle_weights(n))
    return np.asarray(_orthonormal_rows(angle, tol))


def approximation_bias(span: np.ndarray, beta: np.ndarray) -> float:
    """Weight ``E_perp`` of a unit-variance target outside the span of the effects."""
    n = int(round(np.log(beta.size) / np.log(3)))
    coords = np.sqrt(circle_weights(n)) * beta
    coords[0] = 0.0
    total = float(coords @ coords)
    inside = float(np.sum((span @ coords) ** 2))
    return max(total - inside, 0.0) / total


# ---------------------------------------------------------------------------
# Level II: local features, ridge readout, bias and variance
# ---------------------------------------------------------------------------


def _apply_adjacent(psi: np.ndarray, gate: np.ndarray, i: int, n: int) -> np.ndarray:
    """Apply a gate on ``(i, i + 1)`` to a batch of states ``(B, 2^n)``."""
    b = psi.shape[0]
    v = psi.reshape(b, 2**i, 4, 2 ** (n - i - 2))
    return np.matmul(gate, v).reshape(b, -1)


def sample_shots(
    layers: Sequence[Layer],
    n: int,
    p: float,
    theta: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    batch: int = 4096,
) -> np.ndarray:
    """Draw ``shots`` single-shot records of each input, with the Born rule.

    Same features and conventions as
    :func:`QELM.evaluation.fourier_tasks.sample_records` (fresh mask for
    every shot), on a batch of state vectors at once.

    Returns:
        ``int8`` array ``(S, shots, T n + n)``.
    """
    theta = np.atleast_2d(theta)
    s_count = theta.shape[0]
    depth = len(layers)
    width = depth * n + n
    for layer in layers:
        for i, j, _ in layer:
            if j != i + 1:
                raise ValueError("sample_shots needs gates on adjacent sites (i, i+1).")
    inputs = np.repeat(np.arange(s_count), shots)
    out = np.zeros((inputs.size, width), dtype=np.int8)
    states = np.stack([product_state(angles) for angles in theta])
    for start in range(0, inputs.size, batch):
        idx = inputs[start : start + batch]
        b = idx.size
        psi = states[idx]
        for t in range(depth):
            for i, _, gate in layers[t]:
                psi = _apply_adjacent(psi, gate, i, n)
            draws = rng.random((b, n)) < p
            for site in range(n):
                rows = np.flatnonzero(draws[:, site])
                if rows.size:
                    psi[rows], o = _measure(psi[rows], site, n, rng)
                    out[start + rows, t * n + site] = 1 - 2 * o
        for site in range(n):
            psi, o = _measure(psi, site, n, rng)
            out[start : start + b, depth * n + site] = 1 - 2 * o
    return out.reshape(s_count, shots, width)


def _measure(
    psi: np.ndarray, site: int, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Measure ``Z`` of ``site`` on every state of the batch; outcomes 0 or 1."""
    b = psi.shape[0]
    v = psi.reshape(b, 2**site, 2, 2 ** (n - site - 1))
    p1 = np.sum(np.abs(v[:, :, 1, :]) ** 2, axis=(1, 2))
    o = (rng.random(b) < p1).astype(np.int8)
    keep = np.where(o == 1, 1, 0)
    out = np.zeros_like(v)
    rows = np.arange(b)
    out[rows, :, keep, :] = v[rows, :, keep, :]
    norm = np.sqrt(np.where(o == 1, p1, 1.0 - p1))
    out /= norm[:, None, None, None]
    return out.reshape(b, -1), o


def generalized_spectrum(signal: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Squared canonical correlations ``rho^2`` of one shot of the local features.

    Generalised eigenvalues of ``(Sigma, Sigma + Noise)``, decreasing, in
    ``[0, 1]``; directions of the feature space where the record is
    deterministic (zero total variance) are dropped.
    """
    total = signal + noise
    lam, vec = np.linalg.eigh(total)
    keep = lam > 1e-10 * max(float(lam[-1]), 1e-300)
    w = vec[:, keep] / np.sqrt(lam[keep])
    rho2 = np.linalg.eigvalsh(w.T @ signal @ w)
    return np.asarray(np.clip(rho2[::-1], 0.0, 1.0))


@dataclass(frozen=True)
class RidgeSweep:
    """Ridge errors over a grid of ``lambda``, averaged over the test inputs.

    Attributes:
        lambdas: Relative regularisations ``lambda / (S tr(Sigma_hat) / d)``.
        val_mse: Mean squared validation error, mean over the repetitions.
        test_mse: Same on the test inputs.
        bias2: Squared bias on the test inputs (unbiased in the number of
            repetitions).
        variance: Variance over the repetitions (unbiased, ``ddof = 1``),
            zero for a single repetition.
        reps: Number of repetitions.
    """

    lambdas: np.ndarray
    val_mse: np.ndarray
    test_mse: np.ndarray
    bias2: np.ndarray
    variance: np.ndarray
    reps: int


def ridge_sweep(
    train: np.ndarray,
    y_train: np.ndarray,
    val: np.ndarray,
    y_val: np.ndarray,
    test: np.ndarray,
    y_test: np.ndarray,
    lambdas: np.ndarray,
) -> RidgeSweep:
    """Ridge with an unpenalised intercept, for every ``lambda`` and repetition.

    Args:
        train, val, test: Features of shape ``(R, S, d)``: ``R`` repetitions
            (independent shot draws on the same inputs).
        y_train, y_val, y_test: Targets of the inputs, shape ``(S,)``.
        lambdas: Relative regularisations; the absolute one is
            ``lambda * S * tr(Sigma_hat) / d`` for each repetition.
    """
    reps = train.shape[0]
    lambdas = np.asarray(lambdas, dtype=float)
    val_err = np.zeros((reps, lambdas.size))
    preds = np.zeros((reps, lambdas.size, test.shape[1]))
    for r in range(reps):
        mean = train[r].mean(axis=0)
        x = train[r] - mean
        ym = float(y_train.mean())
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        scale = float(np.sum(s**2)) / x.shape[1]
        lam = lambdas * max(scale, 1e-300)
        uty = u.T @ (y_train - ym)
        filt = s[None, :] / (s[None, :] ** 2 + lam[:, None])  # (L, k)
        coef = (filt * uty[None, :]) @ vt  # (L, d)
        pv = (val[r] - mean) @ coef.T + ym
        val_err[r] = np.mean((pv - y_val[:, None]) ** 2, axis=0)
        preds[r] = ((test[r] - mean) @ coef.T + ym).T
    err = preds - y_test[None, None, :]
    test_mse = np.mean(err**2, axis=(0, 2))
    if reps > 1:
        variance = np.mean(np.var(preds, axis=0, ddof=1), axis=1)
    else:
        variance = np.zeros(lambdas.size)
    bias2 = test_mse - variance * (reps - 1) / reps if reps > 1 else test_mse
    return RidgeSweep(
        lambdas=lambdas,
        val_mse=val_err.mean(axis=0),
        test_mse=test_mse,
        bias2=np.asarray(bias2),
        variance=np.asarray(variance),
        reps=reps,
    )


def shot_groups(shots: np.ndarray, k: int, max_reps: int) -> np.ndarray:
    """Average disjoint groups of ``k`` shots: ``(S, K_max, d) -> (R, S, d)``.

    ``R = min(K_max // k, max_reps)`` independent ``k``-shot averages of
    each input.
    """
    reps = min(shots.shape[1] // k, max_reps)
    if reps < 1:
        raise ValueError(f"Need at least {k} shots per input, got {shots.shape[1]}.")
    used = shots[:, : reps * k, :].astype(float)
    grouped = used.reshape(shots.shape[0], reps, k, shots.shape[2]).mean(axis=2)
    return np.asarray(np.moveaxis(grouped, 1, 0))
