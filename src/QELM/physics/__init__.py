"""Physics building blocks.

Readouts (:class:`Readout`) are ways of measuring the state, such as
projective measurement. Observables (:class:`Observables`) are what is
computed from the state, such as ``Tr(Z_i rho)`` on every qubit.
"""

from __future__ import annotations

from QELM.physics.observables import DoubleObservable, Observables, SingleObservable
from QELM.physics.projective import ProjectiveMeasurement, measure_projective_rho
from QELM.physics.readout import Readout, ReadoutResult

_READOUTS: dict[str, type[Readout]] = {
    ProjectiveMeasurement.name: ProjectiveMeasurement,
}


def get_readout(name: str) -> type[Readout]:
    """Return the readout class for a config key.

    Args:
        name: Readout name such as ``"projective_measurement"``.

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
