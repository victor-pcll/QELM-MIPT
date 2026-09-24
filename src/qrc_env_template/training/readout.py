"""Linear readout scaffold."""

from __future__ import annotations

from qrc_env_template.training.base import Trainer


class Readout(Trainer):
    """Reservoir readout. Training is not implemented."""

    name = "readout"
