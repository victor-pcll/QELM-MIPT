"""Strong projective measurement of single qubits in the Z basis.

Measuring qubit ``i`` gives outcome ``m`` in ``{0, 1}`` with the Born
probability ``p_m = Tr(P_m rho)``, where ``P_m`` projects qubit ``i`` on
``|m>``. The state collapses to ``P_m rho P_m / p_m``. Qubit 0 is the
leftmost factor of the tensor product, as in
:func:`QELM.physics.operators.get_site_operator`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from QELM.physics.operators import qubit_counts
from QELM.training.measurements.base import Readout, ReadoutResult


def _check_p(p: Any) -> float:
    """Return ``p`` as a float after checking it is a number in ``[0, 1]``.

    Raises:
        ValueError: If ``p`` is a bool, not a real number, or outside ``[0, 1]``.
    """
    if isinstance(p, bool) or not isinstance(p, Real) or not 0 <= float(p) <= 1:
        raise ValueError(f"p must be a probability in [0, 1], got {p!r}.")
    return float(p)


def measure_projective_rho(
    rho: np.ndarray,
    qubit_idx: int,
    N: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Mesure projective forte stochastique du qubit ``qubit_idx`` de ``rho``.

    Args:
        rho: Density matrix of shape ``(2**N, 2**N)``.
        qubit_idx: Measured qubit, from 0 to ``N - 1``.
        N: Number of qubits. ``None`` reads it from the shape of ``rho``.
        rng: Random generator for the outcome. ``None`` uses the global
            ``numpy.random`` state seeded by :func:`QELM.seed.set_seed`.

    Returns:
        ``(rho_collapsed, mi)``: the post-measurement density matrix of the
        same shape, and the Z eigenvalue of the outcome: ``mi = +1`` for
        ``|0>`` and ``mi = -1`` for ``|1>``, so the mean of ``mi`` is
        ``<Z_i>``.

    Raises:
        ValueError: If ``rho`` is not a qubit density matrix, ``N`` does not
            match it, or ``qubit_idx`` is out of range.
    """
    n, _, _ = qubit_counts(rho)
    if N is not None and N != n:
        raise ValueError(f"N={N} does not match rho of shape {rho.shape}.")
    if not 0 <= qubit_idx < n:
        raise ValueError(f"qubit_idx must be in [0, {n - 1}], got {qubit_idx}.")

    indices = np.arange(2**n)
    shift = n - 1 - qubit_idx
    bits = (indices >> shift) & 1
    mask_0 = bits == 0
    mask_1 = bits == 1

    # 1. Règle de Born : p0 = Tr(P_0 rho).
    p0 = float(np.clip(np.real(np.sum(np.diag(rho)[mask_0])), 0.0, 1.0))
    p1 = 1.0 - p0

    # 2. Tirage aléatoire : outcome 0 avec probabilité p0.
    u = rng.random() if rng is not None else np.random.random()
    outcome = 0 if u < p0 else 1

    # 3. Projection P_m rho P_m / p_m (on annule les blocs croisés 0 <-> 1).
    rho_collapsed = np.zeros_like(rho)
    if outcome == 0:
        rho_collapsed[np.ix_(mask_0, mask_0)] = rho[np.ix_(mask_0, mask_0)] / p0
        mi = 1  # Z|0> = +|0>
    else:
        rho_collapsed[np.ix_(mask_1, mask_1)] = rho[np.ix_(mask_1, mask_1)] / p1
        mi = -1  # Z|1> = -|1>

    return rho_collapsed, mi


def measure_projective_psi(
    psi: np.ndarray,
    qubit_idx: int,
    N: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Measure qubit ``qubit_idx`` of a pure state ``psi`` in Z (issue #183).

    The state-vector counterpart of :func:`measure_projective_rho`: the same
    Born probability ``p0 = sum |psi_b|**2`` over the basis states ``b``
    with the qubit at 0, one uniform number from the same generator, the
    same outcome rule, and the collapse ``P_m psi / sqrt(p_m)``. For
    ``rho = |psi><psi|`` both functions draw the same outcome and return
    ``|psi_m><psi_m| = rho_m``, at ``O(2**N)`` cost instead of
    ``O(4**N)``.

    Args:
        psi: State vector of shape ``(2**N,)``.
        qubit_idx: Measured qubit, from 0 to ``N - 1``.
        N: Number of qubits. ``None`` reads it from the shape of ``psi``.
        rng: Random generator for the outcome. ``None`` uses the global
            ``numpy.random`` state seeded by :func:`QELM.seed.set_seed`.

    Returns:
        ``(psi_collapsed, mi)``: the normalized post-measurement state and
        the Z eigenvalue of the outcome, ``+1`` for ``|0>``, ``-1`` for
        ``|1>``.

    Raises:
        ValueError: If ``psi`` is not a qubit state vector, ``N`` does not
            match it, or ``qubit_idx`` is out of range.
    """
    n = _vector_qubits(psi)
    if N is not None and N != n:
        raise ValueError(f"N={N} does not match psi of shape {psi.shape}.")
    if not 0 <= qubit_idx < n:
        raise ValueError(f"qubit_idx must be in [0, {n - 1}], got {qubit_idx}.")

    bits = (np.arange(2**n) >> (n - 1 - qubit_idx)) & 1
    mask_0 = bits == 0
    weights = np.abs(psi) ** 2
    p0 = float(np.clip(np.sum(weights[mask_0]), 0.0, 1.0))
    p1 = 1.0 - p0

    u = rng.random() if rng is not None else np.random.random()
    outcome = 0 if u < p0 else 1

    psi_collapsed = np.zeros_like(psi)
    if outcome == 0:
        psi_collapsed[mask_0] = psi[mask_0] / np.sqrt(p0)
        mi = 1
    else:
        psi_collapsed[~mask_0] = psi[~mask_0] / np.sqrt(p1)
        mi = -1
    return psi_collapsed, mi


def _vector_qubits(psi: np.ndarray) -> int:
    """Return ``N`` for a state vector of shape ``(2**N,)``."""
    dim = psi.shape[0] if psi.ndim == 1 else 0
    n = dim.bit_length() - 1
    if dim < 2 or 2**n != dim:
        raise ValueError(f"psi must have shape (2**N,), got {psi.shape}.")
    return n


class ProjectiveMeasurement(Readout):
    """Measure qubits one after another, collapsing ``rho`` each time.

    Attributes:
        qubits: Measured qubits in order, from ``params["qubits"]``, or
            ``None`` to measure every system qubit ``Nr .. N-1``.
        Nr: Number of ancilla qubits at the start of the chain (most
            significant bits), from ``params["Nr"]``, 0 by default.
            They are skipped when ``qubits`` is ``None``, so the ancillas
            never collapse; an explicit ``qubits`` list is used as is.
        p: Measurement probability from ``params["p"]``, used by
            :meth:`measure` when no ``p`` is passed; ``None`` if the block
            has no ``p``, in which case ``measure`` uses 1.0.
    """

    name: str = "projective_measurement"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the measured qubits, the probability and the ancilla count.

        Args:
            params: ``qubits`` is an optional list of qubit indices; ``p`` is
                an optional probability in ``[0, 1]``; ``Nr`` is the
                optional number of ancilla qubits at the start of the chain
                (0 when absent).

        Raises:
            ValueError: If ``p`` is given and is not a number in ``[0, 1]``,
                or if ``Nr`` is negative.
        """
        super().__init__(params)
        raw = params.get("qubits")
        self.qubits: list[int] | None = None if raw is None else [int(q) for q in raw]
        raw_p = params.get("p")
        self.p: float | None = None if raw_p is None else _check_p(raw_p)
        self.Nr: int = int(params.get("Nr") or 0)
        if self.Nr < 0:
            raise ValueError(f"Nr must be >= 0, got {self.Nr}.")

    def measure(
        self,
        rho: np.ndarray,
        p: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> ReadoutResult:
        """Measure each target qubit projectively with probability ``p``.

        The targets are ``self.qubits`` when set, otherwise the system qubits
        ``Nr .. N-1``: the ancillas are left unmeasured. Each target
        qubit, in order, is measured with probability ``p`` and left
        untouched otherwise. With ``p = 1`` every target is measured and
        no extra random number is drawn, so the outcomes are the same as
        before ``p`` existed for a given seed. For ``0 < p < 1`` one uniform
        number decides whether a qubit is measured, then a second one (only
        if it is) draws the outcome.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``, or a pure state
                vector of shape ``(2**N,)``, measured with
                :func:`measure_projective_psi` and the same random draws
                (issue #183).
            p: Probability of measuring each target qubit, in ``[0, 1]``.
                ``None`` uses ``self.p`` from the config, or 1.0 if the
                config has no ``p``. A value passed here wins over both.
            rng: Random generator for the choices and outcomes. ``None`` uses
                the global ``numpy.random`` state.

        Returns:
            Outcomes of shape ``(n_targets,)`` as ``float64``: the Z
            eigenvalue ``mi`` in ``{-1, +1}`` for a measured qubit and
            ``np.nan`` for an unmeasured one; and the state after every
            collapse, of the shape of ``rho``.

        Raises:
            ValueError: If ``p`` is not a number in ``[0, 1]``, or if ``rho``
                has fewer qubits than ``Nr``.
        """
        if p is None:
            p = 1.0 if self.p is None else self.p
        p = _check_p(p)
        rho = np.asarray(rho)
        collapse: Callable[..., tuple[np.ndarray, int]]
        if rho.ndim == 1:
            N = _vector_qubits(rho)
            if N < self.Nr:
                raise ValueError(
                    f"Total qubits ({N}) cannot be less than ancilla qubits "
                    f"({self.Nr})."
                )
            collapse = measure_projective_psi
        else:
            N, _, _ = qubit_counts(rho, self.Nr)
            collapse = measure_projective_rho
        qubits = list(range(self.Nr, N)) if self.qubits is None else self.qubits
        outcomes = np.full(len(qubits), np.nan, dtype=np.float64)
        for k, q in enumerate(qubits):
            if p < 1:
                u = rng.random() if rng is not None else np.random.random()
                if u >= p:
                    continue
            rho, outcomes[k] = collapse(rho, q, N, rng)
        return ReadoutResult(outcomes=outcomes, rho=rho)
