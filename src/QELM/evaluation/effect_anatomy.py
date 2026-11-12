"""Anatomy of one mid-circuit record: Kraus operator, effect, Pauli spectrum.

Pedagogical companion of :mod:`QELM.evaluation.effect_mipt` (issue #329).
For one record ``m`` of the mid-circuit measurements of an open brickwork
(convention of :mod:`QELM.evaluation.effect_rank`), this module builds

- the **Kraus operator** ``K_m = P_{m_T} U_T ... P_{m_1} U_1``, the product of
  the gates and of the projectors on the observed outcomes. It acts on the
  state: ``rho -> K_m rho K_m^dagger / Pr(m)``;
- the **effect** ``E_m = K_m^dagger K_m``, which acts on the *input*:
  ``Pr(m | x) = Tr[E_m rho(x)]``. Two Kraus operators that differ by a
  unitary on the left, ``K' = V K``, have the same effect: ``E_m`` keeps what
  the record says about the input and forgets where the state goes next;
- the **Pauli coefficients** ``c_a = Tr[P_a E_m] / 2^N``, so that with a
  product (dense) encoding ``Pr(m | x) = sum_a c_a B_a(x)`` on the Bloch
  monomials, and the **Fourier weight** ``W = sum_{a != 0} c_a^2 / c_0^2``.

The identity behind every figure of ``experiments/effect_anatomy``
(Parseval, ``sum_a c_a^2 = Tr[E^2] / 2^N``)::

    1 + W = 2^N Tr[E^2] / Tr[E]^2 = 2^N / r_2,   r_2 = Tr[E]^2 / Tr[E^2],

with ``r_2`` the participation ratio of the spectrum of ``E_m``
(``1 <= r_2 <= rank``). Hence ``0 <= W <= 2^N - 1``: ``W = 0`` if and only if
``E_m`` is proportional to the identity, and ``W = 2^N - 1`` if and only if
``E_m`` has rank one, whatever the gates. For Clifford gates ``E_m`` is
proportional to a stabilizer projector of ``k`` independent generators: its
``2^(N - k)`` non-zero eigenvalues are equal, so ``r_2 = rank``, and its Pauli
coefficients are ``+-c_0`` on the ``2^k`` strings of the stabilizer group and
zero elsewhere, so ``1 + W = 2^k = 2^N / rank`` = number of non-zero strings.
For Haar gates the non-zero eigenvalues are not equal and the weight is
spread over up to ``4^N`` strings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QELM.evaluation.effect_mipt import apply_gate, project
from QELM.evaluation.effect_rank import brick_pairs

PAULI_LABELS = "IXYZ"

_PAULIS = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def pauli_labels(n: int) -> list[str]:
    """Labels of the ``4^n`` Pauli strings, in the order of :func:`pauli_spectrum`."""
    labels = [""]
    for _ in range(n):
        labels = [s + c for s in labels for c in PAULI_LABELS]
    return labels


def pauli_string(label: str) -> np.ndarray:
    """Matrix of a Pauli string such as ``"XIZ"``, site 0 the most significant bit."""
    out = np.ones((1, 1), dtype=complex)
    for c in label:
        out = np.kron(out, _PAULIS[PAULI_LABELS.index(c)])
    return out


def pauli_spectrum(op: np.ndarray) -> np.ndarray:
    """Coefficients ``c_a = Tr[P_a op] / 2^N`` on the ``4^N`` Pauli strings.

    Args:
        op: Hermitian ``(2**n, 2**n)`` matrix.

    Returns:
        Real array of length ``4**n``, in the order of :func:`pauli_labels`
        (base-4 digits ``I, X, Y, Z = 0, 1, 2, 3``, site 0 first), so that
        ``op = sum_a c_a P_a``.
    """
    dim = op.shape[0]
    n = dim.bit_length() - 1
    # Contract one site at a time: t[a_0, ..., a_{n-1}].
    t = np.asarray(op, dtype=complex).reshape((2,) * (2 * n))
    order = [k for site in range(n) for k in (site, n + site)]
    t = np.ascontiguousarray(t.transpose(order)).reshape((2, 2) * n)
    for _ in range(n):
        # Leading pair of axes (row, col) of the next site -> Pauli index,
        # moved to the end so that the sites keep their order.
        t = np.einsum("pji,ij...->...p", np.stack(_PAULIS), t) / 2
    return np.asarray(t.real).reshape(-1)


def kraus_operator(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    """Kraus operator of the record ``outcomes`` (not normalised).

    Args:
        gates: ``gates[t][g]``, the two-qubit gates of layer ``t``, order of
            :func:`QELM.evaluation.effect_rank.brick_pairs`.
        masks: Boolean ``(T, N)``, the mid-circuit ``Z`` measurements.
        outcomes: Integer ``(T, N)``, the outcome (0 or 1) of each measured
            site; ignored where the mask is ``False``.

    Returns:
        ``K_m`` as a ``(2**n, 2**n)`` matrix; ``Pr(m | rho) = Tr[K rho K^dagger]``.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    dim = 2**n
    pairs = brick_pairs(n)
    # Row j of kt is K |j>, so kt = K^T.
    kt = np.eye(dim, dtype=complex)
    for t in range(depth):
        for site, gate in zip(pairs, gates[t], strict=True):
            kt = apply_gate(kt, n, site, gate)
        for site in np.flatnonzero(masks[t]).tolist():
            kt = project(kt, n, site, int(outcomes[t, site]))
    return np.asarray(kt.T)


def sample_outcomes(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a record with the Born rule for the maximally mixed input.

    Returns:
        Integer ``(T, N)`` array of outcomes, 0 where the mask is ``False``.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    dim = 2**n
    pairs = brick_pairs(n)
    outcomes = np.zeros((depth, n), dtype=int)
    kt = np.eye(dim, dtype=complex) / np.sqrt(dim)
    for t in range(depth):
        for site, gate in zip(pairs, gates[t], strict=True):
            kt = apply_gate(kt, n, site, gate)
        for site in np.flatnonzero(masks[t]).tolist():
            v = kt.reshape(dim, 2**site, 2, 2 ** (n - site - 1))
            p0 = float(np.sum(np.abs(v[:, :, 0, :]) ** 2))
            o = 0 if rng.random() < p0 else 1
            outcomes[t, site] = o
            kt = project(kt, n, site, o) / np.sqrt(p0 if o == 0 else 1 - p0)
    return outcomes


@dataclass(frozen=True)
class EffectAnatomy:
    """Every quantity of one record that the figures show.

    Attributes:
        kraus: ``K_m``.
        effect: ``E_m = K_m^dagger K_m``, rescaled so that ``Tr E = 2^N``
            (``c_0 = 1``; ``E = I`` when nothing is measured).
        prob: ``Pr(m) = Tr[E_m] / 2^N`` for the maximally mixed input,
            before the rescaling.
        eigenvalues: Spectrum of the rescaled ``E_m``, decreasing.
        rank: Number of eigenvalues above ``tol`` times the largest.
        participation: ``r_2 = Tr[E]^2 / Tr[E^2]``.
        coefficients: ``c_a / c_0`` on the ``4^N`` strings.
        n_strings: Number of strings with ``|c_a| > tol c_0``.
        weight: ``W = sum_{a != 0} c_a^2 / c_0^2``.
    """

    kraus: np.ndarray
    effect: np.ndarray
    prob: float
    eigenvalues: np.ndarray
    rank: int
    participation: float
    coefficients: np.ndarray
    n_strings: int
    weight: float


def anatomy(kraus: np.ndarray, tol: float = 1e-9) -> EffectAnatomy:
    """Effect, spectrum, Pauli coefficients and Fourier weight of ``K_m``."""
    effect = kraus.conj().T @ kraus
    effect = (effect + effect.conj().T) / 2
    dim = effect.shape[0]
    trace = float(np.trace(effect).real)
    if trace <= 0:
        raise ValueError("record of zero probability")
    effect = effect * (dim / trace)
    eig = np.sort(np.linalg.eigvalsh(effect))[::-1]
    eig = np.where(np.abs(eig) < tol * eig[0], 0.0, eig)
    coef = pauli_spectrum(effect)
    coef = np.where(np.abs(coef) < tol, 0.0, coef)
    sq = coef**2
    return EffectAnatomy(
        kraus=kraus,
        effect=effect,
        prob=trace / dim,
        eigenvalues=eig,
        rank=int(np.sum(eig > tol * eig[0])),
        participation=float(eig.sum() ** 2 / np.sum(eig**2)),
        coefficients=coef,
        n_strings=int(np.sum(np.abs(coef) > tol)),
        weight=float(sq.sum() - sq[0]),
    )


def product_states(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` product pure states, Bloch vectors uniform on each sphere.

    Returns:
        State vectors, shape ``(count, 2**n)``, site 0 the most significant bit.
    """
    theta = np.arccos(rng.uniform(-1, 1, size=(count, n)))
    phi = rng.uniform(0, 2 * np.pi, size=(count, n))
    amp = np.stack(
        [np.cos(theta / 2) + 0j, np.exp(1j * phi) * np.sin(theta / 2)], axis=-1
    )
    out = amp[:, 0, :]
    for site in range(1, n):
        out = np.einsum("ci,cj->cij", out, amp[:, site, :]).reshape(count, -1)
    return np.asarray(out)


def likelihood_ratio(effect: np.ndarray, states: np.ndarray) -> np.ndarray:
    """``Pr(m | x) / Pr(m)`` for each state, with ``effect`` rescaled to ``Tr = 2^N``.

    It is 1 for every input when ``E_m = I`` (the record says nothing about
    ``x``) and at most ``2^N / rank`` times the largest normalised eigenvalue.
    """
    return np.asarray(np.einsum("ci,ij,cj->c", states.conj(), effect, states).real)
