"""Physics building blocks: operators, readouts and measurements."""

from __future__ import annotations

from QELM.physics.observables import DoubleObservable, Observables, SingleObservable
from QELM.physics.projective import ProjectiveMeasurement, measure_projective_rho
from QELM.physics.readout import Readout, ReadoutResult

_READOUTS: dict[str, type[Readout]] = {
    SingleObservable.name: SingleObservable,
    DoubleObservable.name: DoubleObservable,
    ProjectiveMeasurement.name: ProjectiveMeasurement,
}


def get_readout(name: str) -> type[Readout]:
    """Return the readout class for a config key.

    Args:
        name: Readout name such as ``"single_observable"`` or
            ``"projective_measurement"``.

    Returns:
        The matching :class:`Readout` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _READOUTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_READOUTS))
        raise KeyError(f"Unknown readout {name!r}. Known: {known}.") from exc


__all__ = [
    "DoubleObservable",
    "Observables",
    "ProjectiveMeasurement",
    "Readout",
    "ReadoutResult",
    "SingleObservable",
    "get_readout",
    "measure_projective_rho",
]
