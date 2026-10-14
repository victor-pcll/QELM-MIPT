"""Measurements on the reservoir state, with their back-action.

A measurement scheme (a :class:`Readout`) acts on a density matrix ``rho``
and returns a :class:`ReadoutResult`: the random outcomes and the state
after the measurement back-action. :class:`ProjectiveMeasurement` is the
first one (strong projective measurement in the Z basis); weak measurements
for MIPT or charge sharpening will live here too.

Measuring qubit ``i`` projectively gives outcome ``m`` in ``{0, 1}`` with
the Born probability ``p_m = Tr(P_m rho)``, where ``P_m`` projects qubit
``i`` on ``|m>``. The state collapses to ``P_m rho P_m / p_m``. Qubit 0 is
the leftmost factor of the tensor product, as in
:func:`QELM.physics.operators.get_site_operator`.

Measurements are distinct from the observables of
:mod:`QELM.training.readout`, which say *what* is computed from a state,
such as ``Tr(Z_i rho)`` on every qubit, without disturbing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf

from QELM.physics.operators import n_qubits


@dataclass(frozen=True)
class ReadoutResult:
    """Output of one readout.

    Attributes:
        outcomes: Real array of shape ``(n_outcomes,)``, one entry per
            target qubit; ``np.nan`` marks a qubit that was not measured.
        rho: Density matrix of shape ``(2**N, 2**N)`` after the
            measurement back-action.
    """

    outcomes: np.ndarray
    rho: np.ndarray


class Readout:
    """Base class for measurement schemes configured from a Hydra block.

    Attributes:
        name: Config key for this readout; ``from_config`` reads
            ``cfg[name]``.
        params: Scalar parameters from the matching YAML block.
    """

    name: str = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
        """
        self.params: dict[str, Any] = dict(params)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the readout from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this class.

        Raises:
            TypeError: If the config block is not a mapping.
        """
        node = cfg[cls.name]
        raw = (
            OmegaConf.to_container(node, resolve=True)
            if OmegaConf.is_config(node)
            else node
        )
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        return cls({str(key): value for key, value in raw.items()})

    def measure(
        self,
        rho: np.ndarray,
        p: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> ReadoutResult:
        """Measure ``rho`` and apply the back-action.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.
            p: Probability of measuring each target qubit, in ``[0, 1]``.
            rng: Random generator for the outcomes. ``None`` uses the
                global ``numpy.random`` state seeded by
                :func:`QELM.seed.set_seed`.

        Returns:
            The outcomes and the post-measurement state.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} measure is not implemented.")


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
        mi = 1  # Z|0> = +|0>
    else:
        rho_collapsed[np.ix_(mask_1, mask_1)] = rho[np.ix_(mask_1, mask_1)] / p1
        mi = -1  # Z|1> = -|1>

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
        self,
        rho: np.ndarray,
        p: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> ReadoutResult:
        """Measure each target qubit projectively with probability ``p``.

        Each target qubit, in order, is measured with probability ``p`` and
        left untouched otherwise. With ``p = 1`` every qubit is measured and
        no extra random number is drawn, so the outcomes are the same as
        before ``p`` existed for a given seed. For ``0 < p < 1`` one uniform
        number decides whether a qubit is measured, then a second one (only
        if it is) draws the outcome.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.
            p: Probability of measuring each target qubit, in ``[0, 1]``
                (default 1.0).
            rng: Random generator for the choices and outcomes. ``None`` uses
                the global ``numpy.random`` state.

        Returns:
            Outcomes of shape ``(n_targets,)`` as ``float64``: the Z
            eigenvalue ``mi`` in ``{-1, +1}`` for a measured qubit and
            ``np.nan`` for an unmeasured one; and the state after every
            collapse.

        Raises:
            ValueError: If ``p`` is not a number in ``[0, 1]``.
        """
        if isinstance(p, bool) or not isinstance(p, Real) or not 0 <= p <= 1:
            raise ValueError(f"p must be a probability in [0, 1], got {p!r}.")
        N = n_qubits(rho)
        qubits = list(range(N)) if self.qubits is None else self.qubits
        outcomes = np.full(len(qubits), np.nan, dtype=np.float64)
        for k, q in enumerate(qubits):
            if p < 1:
                u = rng.random() if rng is not None else np.random.random()
                if u >= p:
                    continue
            rho, outcomes[k] = measure_projective_rho(rho, q, N, rng)
        return ReadoutResult(outcomes=outcomes, rho=rho)
