"""Continuous monitoring: Z measurements as a Poisson process (issue #241).

Each qubit is measured in Z at the random times of a Poisson process of rate
``gamma``, independently of the others, while the state evolves under a fixed
Hamiltonian. The time is cut into windows of length ``tau``, each into
``n_sub`` micro-steps ``dt = tau / n_sub``. In each micro-step the state
evolves by ``U(dt) = exp(-i H dt)``, then each qubit is measured with the
probability ``1 - exp(-gamma dt)`` of at least one Poisson event in ``dt``.

The rate is calibrated on the probability ``p`` of the discrete protocols
(issue #241): ``gamma = -ln(1 - p) / tau``, so that a qubit is measured at
least once in a window of length ``tau`` with probability exactly ``p``, as
it is measured with probability ``p`` after each layer of the Floquet chain.
In a micro-step this gives ``1 - (1 - p)**(1 / n_sub)``.

Trajectories are evolved in batches, the columns of a ``(dim, B)`` matrix,
so that ``U(dt)`` is applied by one matrix product; each column has its own
``p``.
"""

from __future__ import annotations

import numpy as np

from QELM.training.measurements.projective import measure_projective_psi


def poisson_rate(p: np.ndarray | float, tau: float) -> np.ndarray:
    """Return ``gamma = -ln(1 - p) / tau``, ``inf`` for ``p = 1``.

    Args:
        p: Probabilities in ``[0, 1]``.
        tau: Window length, ``> 0``.
    """
    p_arr = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return np.asarray(-np.log1p(-p_arr) / tau)


def micro_step_probability(p: np.ndarray | float, n_sub: int) -> np.ndarray:
    """Return ``1 - exp(-gamma dt) = 1 - (1 - p)**(1 / n_sub)``.

    The probability that a qubit is measured in one micro-step, when it is
    measured at least once in a window of ``n_sub`` micro-steps with
    probability ``p``.
    """
    p_arr = np.asarray(p, dtype=float)
    return np.asarray(1.0 - (1.0 - p_arr) ** (1.0 / n_sub))


def poisson_monitored_batch(
    U_dt: np.ndarray,
    psi0: np.ndarray,
    p: np.ndarray,
    n_windows: int,
    n_sub: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a batch of monitored trajectories under the same ``U(dt)``.

    Args:
        U_dt: Unitary of one micro-step, shape ``(dim, dim)``.
        psi0: Initial state of every trajectory, shape ``(dim,)``.
        p: Probability of each trajectory, shape ``(B,)``, that a qubit is
            measured at least once in a window.
        n_windows: Number of windows, the rows of the record.
        n_sub: Micro-steps per window, ``>= 1``.
        rng: Generator of the measurement times and outcomes.

    Returns:
        ``(M, psi)``: the record, shape ``(B, n_windows, N)``, with the last
        outcome (+-1) of each qubit in each window and 0 when it was not
        measured there; and the final states, shape ``(dim, B)``.

    Raises:
        ValueError: If ``n_sub < 1`` or a probability is outside ``[0, 1]``.
    """
    p = np.asarray(p, dtype=float)
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"p must be in [0, 1], got {p}.")
    dim = psi0.shape[0]
    N = dim.bit_length() - 1
    B = p.size
    q = micro_step_probability(p, n_sub)
    psi = np.repeat(np.asarray(psi0, dtype=complex)[:, None], B, axis=1)
    M = np.zeros((B, n_windows, N))
    for j in range(n_windows):
        for _ in range(n_sub):
            psi = U_dt @ psi
            hits = rng.random((B, N)) < q[:, None]
            for b, k in zip(*np.nonzero(hits), strict=True):
                psi[:, b], M[b, j, k] = measure_projective_psi(
                    psi[:, b], int(k), N, rng
                )
    return M, psi
