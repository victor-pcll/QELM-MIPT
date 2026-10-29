"""Two-point correlators ``<A_i B_j>`` for ``i < j``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.physics.operators import get_site_operator, pauli
from QELM.training.readout.base import Observables, _labels


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

    def operators(self, n_qubits: int | None = None) -> list[np.ndarray]:
        """Return ``A_i B_j`` for each pair, then each ``i < j``.

        That is ``n (n - 1) / 2`` features per pair: ``(0, 1), (0, 2), ...,
        (n-2, n-1)``.

        Args:
            n_qubits: Number of qubits of the state, ancillas included.
                ``None`` uses ``self.n_qubits`` from the config.

        Raises:
            ValueError: If ``n_qubits`` is ``None`` and the config has none.
        """
        n = self._resolve_n_qubits(n_qubits)
        ops = []
        for a, b in self.pairs:
            A, B = pauli(a), pauli(b)
            ops += [
                get_site_operator(A, i, n) @ get_site_operator(B, j, n)
                for i in range(n)
                for j in range(i + 1, n)
            ]
        return ops
