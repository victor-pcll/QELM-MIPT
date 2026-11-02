"""Feature extraction: observables measured on the final reservoir state.

Each single observable ``O`` gives the ``n`` features ``<O_i> = Tr(rho O_i)``.
Each double observable ``(A, B)`` gives the ``n (n - 1) / 2`` two-point
correlators ``<A_i B_j>`` for ``i < j``. ``n`` is the number of qubits of
the state the readout acts on, ancillas included: ``n_qubits``, the
top-level ``N_total = N + Nr`` in ``conf/`` (issue #235). Operators are Pauli labels
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
        n_qubits: Number of qubits of the state, ancillas included, from
            ``params["n_qubits"]``, used by :meth:`operators` when no
            ``n_qubits`` is passed; ``None`` if the block has none.
    """

    name: str = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``n_qubits`` (number of qubits of the state, ancillas
                included) is optional.
        """
        self.params: dict[str, Any] = dict(params)
        raw_n = params.get("n_qubits")
        self.n_qubits: int | None = None if raw_n is None else int(raw_n)

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

    def _resolve_n_qubits(self, n_qubits: int | None) -> int:
        """Return ``n_qubits`` if given, else ``self.n_qubits``.

        Raises:
            ValueError: If both are ``None``.
        """
        if n_qubits is not None:
            return n_qubits
        if self.n_qubits is None:
            raise ValueError(
                f"{self.name}: n_qubits was not passed and the config block has "
                f"no n_qubits. Pass operators(n_qubits=...) or add n_qubits to "
                f"the {self.name} block."
            )
        return self.n_qubits

    def operators(self, n_qubits: int | None = None) -> list[np.ndarray]:
        """Return the measured operators on ``n_qubits`` qubits.

        Args:
            n_qubits: Number of qubits of the state. ``None`` uses
                ``self.n_qubits`` from the config.

        Returns:
            Operators of shape ``(2**n_qubits, 2**n_qubits)``, one per
            feature.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} operators are not implemented.")

    def expectation(self, rho: np.ndarray) -> np.ndarray:
        """Return the features ``Tr(rho O_k)`` for every operator ``O_k``.

        Args:
            rho: Density matrix of shape ``(2**n, 2**n)``.

        Returns:
            Real array of shape ``(n_features,)``.
        """
        ops = self.operators(qubit_counts(rho).N_total)
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
