"""Feature extraction: observables measured on the final reservoir state.

Each single observable ``O`` gives the ``N`` features ``<O_i> = Tr(rho O_i)``.
Each double observable ``(A, B)`` gives the ``N (N - 1) / 2`` two-point
correlators ``<A_i B_j>`` for ``i < j``. Operators are Pauli labels
(``"X"``, ``"Y"`` or ``"Z"``) from YAML. A set can hold several observables
(or pairs), or none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf

from QELM.physics.operators import pauli, qubit_counts


class Observables:
    """Set of operators whose expectation values are the QELM features.

    Attributes:
        name: Config key for this observable set; ``from_config`` reads
            ``cfg[name]``.
        params: Scalar parameters from the matching YAML block.
        N: Number of qubits from ``params["N"]``, used by :meth:`operators`
            when no ``N`` is passed; ``None`` if the block has no ``N``.
    """

    name: str = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``N`` (number of qubits) is optional.
        """
        self.params: dict[str, Any] = dict(params)
        raw_N = params.get("N")
        self.N: int | None = None if raw_N is None else int(raw_N)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the observable set from the Hydra block named ``cls.name``.

        A missing or empty block gives a set without observables, whose
        :meth:`expectation` returns no features.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                parameters, or is absent.

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

    def _resolve_N(self, N: int | None) -> int:
        """Return ``N`` if given, else ``self.N``.

        Raises:
            ValueError: If both are ``None``.
        """
        if N is not None:
            return N
        if self.N is None:
            raise ValueError(
                f"{self.name}: N was not passed and the config block has no N. "
                f"Pass operators(N=...) or add N to the {self.name} block."
            )
        return self.N

    def operators(self, N: int | None = None) -> list[np.ndarray]:
        """Return the measured operators on ``N`` qubits.

        Args:
            N: Number of qubits. ``None`` uses ``self.N`` from the config.

        Returns:
            Operators of shape ``(2**N, 2**N)``, one per feature.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} operators are not implemented.")

    def expectation(self, rho: np.ndarray) -> np.ndarray:
        """Return the features ``Tr(rho O_k)`` for every operator ``O_k``.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.

        Returns:
            Real array of shape ``(n_features,)``.
        """
        N, _, _ = qubit_counts(rho)
        ops = self.operators(N)
        # Tr(rho O) = sum_ij rho_ij O_ji; real because rho and O are Hermitian.
        return np.array([np.einsum("ij,ji->", rho, op).real for op in ops])


def _labels(raw: Any, key: str) -> list[str]:
    """Return a list of checked Pauli labels from a YAML value.

    Args:
        raw: A label such as ``"Z"``, a list of labels, or ``None``.
        key: Parameter name, for the error message.

    Returns:
        The labels, possibly empty.

    Raises:
        ValueError: If a label is not a Pauli label.
        TypeError: If ``raw`` is neither a string nor a list.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list | tuple):
        raise TypeError(f"{key} must be a Pauli label or a list of them, got {raw!r}.")
    labels = [str(label) for label in raw]
    for label in labels:
        pauli(label)
    return labels
