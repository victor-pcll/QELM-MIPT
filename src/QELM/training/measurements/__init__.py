"""Measurement schemes: the :class:`Readout` parent and one module per scheme.

``projective.py``: strong Z measurement with probability ``p``;
``weak.py``: weak charge measurement of strength ``gamma`` (issue #137).
"""

from __future__ import annotations

from QELM.training.measurements.base import Readout, ReadoutResult
from QELM.training.measurements.projective import (
    ProjectiveMeasurement,
    measure_projective_psi,
    measure_projective_rho,
)
from QELM.training.measurements.weak import WeakMeasurement, measure_weak, weak_kraus

__all__ = [
    "ProjectiveMeasurement",
    "Readout",
    "ReadoutResult",
    "WeakMeasurement",
    "measure_projective_psi",
    "measure_projective_rho",
    "measure_weak",
    "weak_kraus",
]
