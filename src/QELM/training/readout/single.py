"""One-site observables ``<O_i>``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.physics.operators import get_site_operator, pauli
from QELM.training.readout.base import Observables, _labels


class SingleObservable(Observables):
    """One-site expectation values ``<O_i>`` for each ``O`` and ``i = 0 .. n-1``.

    YAML: ``single_observable: {observables: [Z, X]}``. The legacy form
    ``{observable: Z}`` is still read. Features come observable by
    observable: ``<O_0>, ..., <O_{n-1}>`` for the first, then the next.

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

    def operators(self, n_qubits: int | None = None) -> list[np.ndarray]:
        """Return ``O_i`` for each observable, then each site: ``len * n`` features.

        Args:
            n_qubits: Number of qubits of the state, ancillas included.
                ``None`` uses ``self.n_qubits`` from the config.

        Raises:
            ValueError: If ``n_qubits`` is ``None`` and the config has none.
        """
        n = self._resolve_n_qubits(n_qubits)
        return [
            get_site_operator(pauli(label), i, n)
            for label in self.observables
            for i in range(n)
        ]
