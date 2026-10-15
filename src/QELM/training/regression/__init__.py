"""Readout fitting: the :class:`Trainer` parent and one module per regression."""

from __future__ import annotations

from QELM.training.regression.base import Trainer
from QELM.training.regression.linear import LinearReadout
from QELM.training.regression.ridge import RidgeRegression

__all__ = ["LinearReadout", "RidgeRegression", "Trainer"]
