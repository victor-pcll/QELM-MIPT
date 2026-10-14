"""Common interface for the ways of measuring the reservoir state.

A readout is a measurement scheme: projective measurement
(:mod:`QELM.physics.projective`), and later weak measurement. It acts on a
density matrix ``rho`` and returns a :class:`ReadoutResult`: the random
outcomes and the state after the measurement back-action.

Readouts are distinct from observables (:mod:`QELM.physics.observables`),
which say *what* is computed from a state, such as ``Tr(Z_i rho)`` on every
qubit, without disturbing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ReadoutResult:
    """Output of one readout.

    Attributes:
        outcomes: Real array of shape ``(n_outcomes,)``, one entry per
            measurement performed.
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
        self, rho: np.ndarray, rng: np.random.Generator | None = None
    ) -> ReadoutResult:
        """Measure ``rho`` and apply the back-action.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.
            rng: Random generator for the outcomes. ``None`` uses the
                global ``numpy.random`` state seeded by
                :func:`QELM.seed.set_seed`.

        Returns:
            The outcomes and the post-measurement state.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} measure is not implemented.")
