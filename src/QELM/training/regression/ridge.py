"""Ridge regression scaffold."""

from __future__ import annotations

from QELM.training.regression.base import Trainer


class RidgeRegression(Trainer):
    """Ridge regression readout.

    ``fit`` accepts features of shape ``(n_samples, n_features)`` and does
    not estimate a weight vector. Selected with ``training=ridge``.
    """

    name = "ridge"
