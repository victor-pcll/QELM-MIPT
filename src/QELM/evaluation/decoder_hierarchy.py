"""Three decoders of the charge on the same U(1) trajectories (issue #307).

Agrawal et al. (PRX 2024, Appendix B) rank the decoders of the charge of a
monitored U(1) circuit: the optimal decoder that knows the gates
("postBQP"), the stat-mech decoder that averages over them (a symmetric
exclusion process, SEP), and a trained network. This module provides the
pieces needed to score them, and the linear QELM readouts, on the same
trajectories of :func:`QELM.evaluation.charge_variance.sample_trajectories`.

The chain starts in ``(|Q0> + |Q1>) / sqrt(2)``. The gates conserve the
charge and the measurements are diagonal, so the final state is
``sqrt(w) |psi_0> + sqrt(1 - w) |psi_1>`` and the records have the law of
a charge drawn with probability 1/2 each. ``w`` is then exactly the
posterior ``P(Q0 | m, U)`` of an observer who knows the gates
(:func:`sector_weight`): the ``postBQP`` decoder of the paper, computed
along the simulation at no extra cost.

Every decoder is scored by its expected accuracy given ``w``
(:func:`expected_accuracy`): a guess ``Q0`` is right with probability ``w``.
This removes the noise of drawing a label from each trajectory; the labels
drawn with :func:`draw_labels` are only used to train the readouts, as an
experiment would (it sees the charge, not ``w``).
"""

from __future__ import annotations

import numpy as np

from QELM.evaluation.learnability import UNMEASURED, site_bits


def sector_weight(probs: np.ndarray, Q: int) -> np.ndarray:
    """Return the weight of each state in the sector of charge ``Q``.

    Args:
        probs: Born distributions ``|psi|^2``, shape ``(n, 2**L)``.
        Q: Charge, number of 1 bits.

    Returns:
        Shape ``(n,)``.
    """
    probs = np.asarray(probs, dtype=float)
    L = probs.shape[1].bit_length() - 1
    in_sector = site_bits(L).sum(axis=1) == int(Q)
    return np.asarray(probs[:, in_sector].sum(axis=1))


def record_features(outcomes: np.ndarray) -> np.ndarray:
    """Return ``vec(M)``: +1 for outcome 0, -1 for outcome 1, 0 if unmeasured.

    The encoding of the QELM-M: the ``Z`` eigenvalue of each measured site,
    and 0 where no measurement happened.

    Args:
        outcomes: Records of shape ``(n, rounds, L)`` (0, 1 or ``UNMEASURED``).

    Returns:
        Float array of shape ``(n, rounds * L)``.
    """
    m = np.asarray(outcomes)
    x = np.where(m == UNMEASURED, 0.0, 1.0 - 2.0 * m)
    return np.asarray(x.reshape(len(m), -1), dtype=float)


def expected_accuracy(w: np.ndarray, guess_q0: np.ndarray) -> np.ndarray:
    """Return the probability that each guess is right, given ``w = P(Q0)``.

    Args:
        w: Posterior of ``Q0`` with the gates known, shape ``(n,)``.
        guess_q0: Guess of each decoder: 1 for ``Q0``, 0 for ``Q1``, 0.5
            for a tie (a coin flip).

    Returns:
        Shape ``(n,)``: ``g w + (1 - g) (1 - w)``.
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(guess_q0, dtype=float)
    return np.asarray(g * w + (1.0 - g) * (1.0 - w))


def guess_from_posterior(post_q0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return the guess ``1`` (``Q0``), ``0`` (``Q1``) or ``0.5`` (tie)."""
    d = np.asarray(post_q0, dtype=float) - 0.5
    return np.where(d > tol, 1.0, np.where(d < -tol, 0.0, 0.5))


def draw_labels(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw the charge of each trajectory, 1 for ``Q0`` with probability ``w``.

    It is the outcome of a final measurement of the charge, the label an
    experiment would train on.
    """
    w = np.asarray(w, dtype=float)
    return np.asarray((rng.random(w.shape) < w).astype(int))


def bayes_share(acc: np.ndarray, acc_bayes: np.ndarray) -> np.ndarray:
    """Return ``g = (acc - 1/2) / (acc_bayes - 1/2)``, NaN where Bayes is at 1/2."""
    acc = np.asarray(acc, dtype=float)
    den = np.asarray(acc_bayes, dtype=float) - 0.5
    return np.asarray(
        np.divide(acc - 0.5, den, out=np.full_like(acc, np.nan), where=den > 1e-3)
    )
