"""Base class of the measurement schemes, with their back-action.

A measurement scheme (a :class:`Readout`) acts on a density matrix ``rho``
and returns a :class:`ReadoutResult`: the random outcomes and the state
after the measurement back-action. Each scheme is a child class in its own
module of :mod:`QELM.training.measurements` (projective measurement today,
weak measurements for MIPT or charge sharpening later).

Measurements are distinct from the observables of
:mod:`QELM.training.readout`, which say *what* is computed from a state,
such as ``Tr(Z_i rho)`` on every qubit, without disturbing it.
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

        Every readout parameter has a default, so a missing or ``null``
        block builds the readout with its defaults.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters, or is absent.

        Returns:
            An instance of this class.

        Raises:
            TypeError: If the config block is not a mapping.
        """
        node = OmegaConf.select(cfg, cls.name)
        if node is None:
            return cls({})
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
