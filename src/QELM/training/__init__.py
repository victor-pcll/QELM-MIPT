"""QELM data pipeline: measurements, feature extraction and readout fitting.

One subpackage per domain, each with a ``base.py`` holding the parent class
and one module per child class:

- :mod:`QELM.training.measurements`: measurement schemes with back-action
  (``base.py``: :class:`Readout`; ``projective.py``:
  :class:`ProjectiveMeasurement`; ``weak.py``: :class:`WeakMeasurement`).
- :mod:`QELM.training.readout`: features computed from the final state
  (``base.py``: :class:`Observables`; ``single.py``:
  :class:`SingleObservable`; ``double.py``: :class:`DoubleObservable`).
- :mod:`QELM.training.dataset`: the dataset ``(Phi, y)`` of measured
  trajectories (:func:`create_qelm_dataset`, :func:`split_dataset`).
- :mod:`QELM.training.regression`: trainers fitted on those features
  (``base.py``: :class:`Trainer`; ``linear.py``: :class:`LinearReadout`;
  ``ridge.py``: :class:`RidgeReadout`; ``pinv.py``:
  :class:`PseudoInverseReadout`).

Every class is importable from :mod:`QELM.training` directly.
"""

from __future__ import annotations

from QELM.training.dataset import QELMDataset, create_qelm_dataset, split_dataset
from QELM.training.measurements import (
    ProjectiveMeasurement,
    Readout,
    ReadoutResult,
    WeakMeasurement,
    measure_projective_rho,
)
from QELM.training.readout import DoubleObservable, Observables, SingleObservable
from QELM.training.regression import (
    LinearReadout,
    PseudoInverseReadout,
    RidgeReadout,
    Trainer,
)

_TRAINERS: dict[str, type[Trainer]] = {
    LinearReadout.name: LinearReadout,
    PseudoInverseReadout.name: PseudoInverseReadout,
    RidgeReadout.name: RidgeReadout,
}

_READOUTS: dict[str, type[Readout]] = {
    ProjectiveMeasurement.name: ProjectiveMeasurement,
    WeakMeasurement.name: WeakMeasurement,
}


def get_trainer(name: str) -> type[Trainer]:
    """Return the trainer class for a ``training.name`` config value.

    Args:
        name: Scalar trainer name such as ``"ridge"`` or ``"linear"``.

    Returns:
        The matching :class:`Trainer` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _TRAINERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TRAINERS))
        raise KeyError(f"Unknown training {name!r}. Known: {known}.") from exc


def get_readout(name: str) -> type[Readout]:
    """Return the measurement scheme for a config key.

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
    "LinearReadout",
    "Observables",
    "QELMDataset",
    "ProjectiveMeasurement",
    "PseudoInverseReadout",
    "Readout",
    "ReadoutResult",
    "RidgeReadout",
    "SingleObservable",
    "Trainer",
    "WeakMeasurement",
    "create_qelm_dataset",
    "get_readout",
    "get_trainer",
    "measure_projective_rho",
    "split_dataset",
]
