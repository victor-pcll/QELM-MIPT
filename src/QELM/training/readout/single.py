"""One-site observables ``<O_i>``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.physics.operators import get_site_operator, pauli
from QELM.training.readout.base import Observables, _labels


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

    def operators(self, N: int | None = None) -> list[np.ndarray]:
        """Return ``O_i`` for each observable, then each site: ``len * N`` features.

        Args:
            N: Number of qubits. ``None`` uses ``self.N`` from the config.

        Raises:
            ValueError: If ``N`` is ``None`` and the config has no ``N``.
        """
        N = self._resolve_N(N)
        return [
            get_site_operator(pauli(label), i, N)
            for label in self.observables
            for i in range(N)
        ]
