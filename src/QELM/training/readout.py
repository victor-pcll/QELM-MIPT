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

from QELM.physics.operators import get_site_operator, n_qubits, pauli


class Observables:
    """Set of operators whose expectation values are the QELM features.

    Attributes:
        name: Config key for this observable set; ``from_config`` reads
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

    def operators(self, N: int) -> list[np.ndarray]:
        """Return the measured operators on ``N`` qubits.

        Args:
            N: Number of qubits.

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
        N, _, _ = n_qubits(rho)
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


class SingleObservable(Observables):
    """One-site expectation values ``<O_i>`` for each ``O`` and ``i = 0 .. N-1``.

    YAML: ``single_observable: {observables: [Z, X]}``. The legacy form
    ``{observable: Z}`` is still read. Features come observable by
    observable: ``<O_0>, ..., <O_{N-1}>`` for the first, then the next.

    Attributes:
        observables: Pauli labels, possibly empty.
    """

    name: str = "single_observable"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store and check the Pauli labels.

        Args:
            params: ``observables`` (a label or a list of labels) and/or the
                legacy ``observable``. Both are optional.
        """
        super().__init__(params)
        self.observables: list[str] = _labels(
            params.get("observable"), "observable"
        ) + _labels(params.get("observables"), "observables")

    def operators(self, N: int) -> list[np.ndarray]:
        """Return ``O_i`` for each observable, then each site: ``len * N`` features."""
        return [
            get_site_operator(pauli(label), i, N)
            for label in self.observables
            for i in range(N)
        ]


class DoubleObservable(Observables):
    """Two-point correlators ``<A_i B_j>`` for each pair ``(A, B)`` and ``i < j``.

    YAML: ``double_observable: {pairs: [[Z, Z], [X, X]]}``. The legacy form
    ``{observable_1: Z, observable_2: Z}`` is still read. Features come pair
    by pair, each in lexicographic site order.

    Attributes:
        pairs: ``(A, B)`` Pauli label pairs, possibly empty.
    """

    name: str = "double_observable"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store and check the Pauli label pairs.

        Args:
            params: ``pairs`` (a list of two-label lists) and/or the legacy
                ``observable_1`` and ``observable_2``. All are optional.

        Raises:
            ValueError: If a pair does not have two labels, or only one of
                ``observable_1`` / ``observable_2`` is given.
        """
        super().__init__(params)
        self.pairs: list[tuple[str, str]] = []
        first, second = params.get("observable_1"), params.get("observable_2")
        if (first is None) != (second is None):
            raise ValueError("observable_1 and observable_2 must be given together.")
        if first is not None:
            self.pairs.append((str(first), str(second)))
        for pair in params.get("pairs") or []:
            labels = _labels(list(pair), "pairs")
            if len(labels) != 2:
                raise ValueError(f"Each pair must have two labels, got {pair!r}.")
            self.pairs.append((labels[0], labels[1]))
        for a, b in self.pairs:
            pauli(a)
            pauli(b)

    def operators(self, N: int) -> list[np.ndarray]:
        """Return ``A_i B_j`` for each pair, then each ``i < j``.

        That is ``N (N - 1) / 2`` features per pair: ``(0, 1), (0, 2), ...,
        (N-2, N-1)``.
        """
        ops = []
        for a, b in self.pairs:
            A, B = pauli(a), pauli(b)
            ops += [
                get_site_operator(A, i, N) @ get_site_operator(B, j, N)
                for i in range(N)
                for j in range(i + 1, N)
            ]
        return ops
