"""Strong projective measurements of single qubits in the Z basis.

Measuring qubit ``i`` gives outcome ``m`` in ``{0, 1}`` with the Born
probability ``p_m = Tr(P_m rho)``, where ``P_m`` projects qubit ``i`` on
``|m>``. The state collapses to ``P_m rho P_m / p_m``. Qubit 0 is the
leftmost factor of the tensor product, as in
:func:`QELM.physics.operators.get_site_operator`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.physics.observables import n_qubits
from QELM.physics.readout import Readout, ReadoutResult


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
        same shape, and ``mi = -1`` for outcome 0, ``mi = +1`` for outcome 1.

    Raises:
        ValueError: If ``rho`` is not a qubit density matrix, ``N`` does not
            match it, or ``qubit_idx`` is out of range.
    """
    n = n_qubits(rho)
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
        mi = -1
    else:
        rho_collapsed[np.ix_(mask_1, mask_1)] = rho[np.ix_(mask_1, mask_1)] / p1
        mi = 1

    return rho_collapsed, mi


class ProjectiveMeasurement(Readout):
    """Measure qubits one after another, collapsing ``rho`` each time.

    Attributes:
        qubits: Measured qubits in order, from ``params["qubits"]``, or
            ``None`` to measure every qubit ``0 .. N-1``.
    """

    name: str = "projective_measurement"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the measured qubits.

        Args:
            params: ``qubits`` is an optional list of qubit indices.
        """
        super().__init__(params)
        raw = params.get("qubits")
        self.qubits: list[int] | None = None if raw is None else [int(q) for q in raw]

    def measure(
        self, rho: np.ndarray, rng: np.random.Generator | None = None
    ) -> ReadoutResult:
        """Measure each qubit with :func:`measure_projective_rho`.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.
            rng: Random generator for the outcomes. ``None`` uses the global
                ``numpy.random`` state.

        Returns:
            Outcomes ``mi`` in ``{-1, +1}`` of shape ``(n_measured,)`` as
            ``float64``, and the state after every collapse.
        """
        N = n_qubits(rho)
        qubits = list(range(N)) if self.qubits is None else self.qubits
        outcomes = np.empty(len(qubits), dtype=np.float64)
        for k, q in enumerate(qubits):
            rho, outcomes[k] = measure_projective_rho(rho, q, N, rng)
        return ReadoutResult(outcomes=outcomes, rho=rho)
