"""Linear readout scaffold."""

from __future__ import annotations

from QELM.training.base import Observables


class SingleObservable(Observables):
    """Scaffold for a single observable readout.

    Attributes:
        name: Config key for this readout. Subclasses override it with their
            ``readout_type`` value; ``from_config`` reads ``cfg[name]``.
    """

    name: str = "single_observable"

    def __init__(self, params: dict[str, float]) -> None:
        """Store the observable's scalar parameter(s).

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``observable`` is required.
        """
        self.observable: float = float(params["observable"])


class DoubleObservable(Observables):
    """Scaffold for a double observable readout.

    Attributes:
        name: Config key for this readout. Subclasses override it with their
            ``readout_type`` value; ``from_config`` reads ``cfg[name]``.
    """

    name: str = "double_observable"

    def __init__(self, params: dict[str, float]) -> None:
        """Store the observable's scalar parameter(s).

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``observable_1`` and ``observable_2`` are required.
        """
        self.observable_1: float = float(params["observable_1"])
        self.observable_2: float = float(params["observable_2"])
