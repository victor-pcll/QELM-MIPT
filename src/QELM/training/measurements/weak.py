"""Weak measurement of the charge of single qubits, of strength ``gamma``.

The scheme of Agrawal et al., PRX 14, 041012 (2024) (issue #137): the
measured qubit controls a rotation ``exp(-i gamma pi X / 2)`` of a fresh
detector qubit in ``|0>``, and the detector is then read out projectively.
The detector clicks (outcome ``m = 1``) only if the qubit holds a charge,
with probability ``sin(gamma pi / 2)**2``. On the measured qubit this is
the pair of Kraus operators, diagonal in the Z basis::

    K_0 = |0><0| + cos(gamma pi / 2) |1><1|      (no click)
    K_1 =          sin(gamma pi / 2) |1><1|      (click)

with ``K_0^dagger K_0 + K_1^dagger K_1 = 1``. ``gamma = 0`` learns nothing
and leaves the state unchanged; ``gamma = 1`` is the projective measurement
of the charge (``K_0 = |0><0|``, ``K_1 = |1><1|``). A click always
projects the qubit on ``|1>``; no click only makes ``|0>`` more likely.

Outcome ``m`` has the Born probability ``p_m = <psi| K_m^dagger K_m |psi>``
and the state becomes ``K_m psi / sqrt(p_m)`` (or ``K_m rho K_m^dagger /
p_m``), so a pure state stays pure: trajectories can be simulated with
state vectors, much cheaper than density matrices.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from QELM.training.measurements.base import Readout, ReadoutResult


def _check_gamma(gamma: Any) -> float:
    """Return ``gamma`` as a float after checking it is a number in ``[0, 1]``.

    Raises:
        ValueError: If ``gamma`` is a bool, not a real number, or outside
            ``[0, 1]``.
    """
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, Real)
        or not 0 <= float(gamma) <= 1
    ):
        raise ValueError(f"gamma must be a strength in [0, 1], got {gamma!r}.")
    return float(gamma)


def weak_kraus(gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the Kraus operators ``(K_0, K_1)`` of strength ``gamma``.

    Args:
        gamma: Measurement strength in ``[0, 1]``.

    Returns:
        Two real diagonal ``2 x 2`` matrices; ``K_1`` is the click.

    Raises:
        ValueError: If ``gamma`` is not in ``[0, 1]``.
    """
    angle = _check_gamma(gamma) * np.pi / 2
    K0 = np.diag([1.0, np.cos(angle)])
    K1 = np.diag([0.0, np.sin(angle)])
    return K0, K1


def measure_weak(
    state: np.ndarray,
    qubit_idx: int,
    gamma: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Weakly measure the charge of qubit ``qubit_idx`` and apply the back-action.

    Both Kraus operators are diagonal, so they act as a rescaling of the
    amplitudes whose bit ``qubit_idx`` is 1: no matrix is built.

    Args:
        state: State vector of shape ``(2**N,)`` or density matrix of shape
            ``(2**N, 2**N)``.
        qubit_idx: Measured qubit, from 0 to ``N - 1``.
        gamma: Measurement strength in ``[0, 1]``.
        rng: Random generator for the outcome. ``None`` uses the global
            ``numpy.random`` state seeded by :func:`QELM.seed.set_seed`.

    Returns:
        ``(state_after, m)``: the normalized post-measurement state, with the
        shape of ``state``, and the outcome ``m = 1`` (click) or ``0``.

    Raises:
        ValueError: If ``state`` is not a qubit state, ``qubit_idx`` is out
            of range or ``gamma`` is not in ``[0, 1]``.
    """
    state = np.asarray(state, dtype=complex)
    dim = state.shape[0]
    N = dim.bit_length() - 1
    if dim < 2 or 2**N != dim or state.shape not in ((dim,), (dim, dim)):
        raise ValueError(
            "state must be a vector (2**N,) or a density matrix (2**N, 2**N), "
            f"got shape {state.shape}."
        )
    if not 0 <= qubit_idx < N:
        raise ValueError(f"qubit_idx must be in [0, {N - 1}], got {qubit_idx}.")
    angle = _check_gamma(gamma) * np.pi / 2

    ones = ((np.arange(dim) >> (N - 1 - qubit_idx)) & 1).astype(bool)
    # Probability that the qubit holds a charge, <n_q> = Tr(|1><1|_q rho).
    if state.ndim == 1:
        n_q = float(np.sum(np.abs(state[ones]) ** 2))
    else:
        n_q = float(np.real(np.sum(np.diag(state)[ones])))
    p1 = float(np.clip(np.sin(angle) ** 2 * n_q, 0.0, 1.0))

    u = rng.random() if rng is not None else np.random.random()
    m = 1 if u < p1 else 0
    # Diagonal of K_m on the full register, then K_m psi or K_m rho K_m.
    k = np.where(ones, np.sin(angle), 0.0) if m else np.where(ones, np.cos(angle), 1.0)
    p_m = p1 if m else 1.0 - p1
    if state.ndim == 1:
        out = k * state / np.sqrt(p_m)
    else:
        out = np.outer(k, k) * state / p_m
    return out, m


class WeakMeasurement(Readout):
    """Weakly measure qubits one after another, with back-action.

    Attributes:
        gamma: Strength from ``params["gamma"]``, used by :meth:`measure`
            when no ``gamma`` is passed; 1.0 (projective) if absent.
        qubits: Measured qubits in order, from ``params["qubits"]``, or
            ``None`` for every system qubit ``Nr .. N-1``.
        Nr: Number of ancilla qubits at the start of the chain, never
            measured when ``qubits`` is ``None``; 0 by default.
    """

    name: str = "weak_measurement"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the strength, the measured qubits and the ancilla count.

        Args:
            params: Optional ``gamma`` in ``[0, 1]``, ``qubits`` (list of
                indices) and ``Nr`` (ancilla count).

        Raises:
            ValueError: If ``gamma`` is not in ``[0, 1]`` or ``Nr`` is
                negative.
        """
        super().__init__(params)
        raw_gamma = params.get("gamma")
        self.gamma: float = 1.0 if raw_gamma is None else _check_gamma(raw_gamma)
        raw = params.get("qubits")
        self.qubits: list[int] | None = None if raw is None else [int(q) for q in raw]
        self.Nr: int = int(params.get("Nr") or 0)
        if self.Nr < 0:
            raise ValueError(f"Nr must be >= 0, got {self.Nr}.")

    def targets(self, N: int) -> list[int]:
        """Return the measured qubits for an ``N``-qubit state."""
        return list(range(self.Nr, N)) if self.qubits is None else self.qubits

    def measure(
        self,
        rho: np.ndarray,
        p: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> ReadoutResult:
        """Weakly measure every target qubit once, in order.

        Args:
            rho: State vector ``(2**N,)`` or density matrix ``(2**N, 2**N)``.
            p: Strength ``gamma`` for this call (the name follows
                :meth:`Readout.measure`). ``None`` uses ``self.gamma``.
            rng: Random generator for the outcomes. ``None`` uses the global
                ``numpy.random`` state.

        Returns:
            The outcomes ``m`` in ``{0, 1}`` (1 = click, the qubit was found
            charged) as ``float64``, one per target, and the state after
            every back-action, with the shape of ``rho``.

        Raises:
            ValueError: If the strength is not in ``[0, 1]``.
        """
        gamma = self.gamma if p is None else _check_gamma(p)
        state = np.asarray(rho, dtype=complex)
        N = state.shape[0].bit_length() - 1
        if N < self.Nr:
            raise ValueError(f"State has N={N} qubits, fewer than Nr={self.Nr}.")
        targets = self.targets(N)
        outcomes = np.zeros(len(targets), dtype=np.float64)
        for k, q in enumerate(targets):
            state, outcomes[k] = measure_weak(state, q, gamma, rng)
        return ReadoutResult(outcomes=outcomes, rho=state)
