"""Linear readout scaffold."""

from __future__ import annotations

from QELM.training.regression.base import Trainer


class LinearReadout(Trainer):
    """Linear readout trained on the reservoir features.

    ``fit`` accepts features of shape ``(n_samples, n_features)`` and does
    not estimate a weight vector. Selected with ``training=readout``.
    """

    name = "readout"
