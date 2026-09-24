"""Kitaev model scaffold."""

from __future__ import annotations

from qrc_env_template.models.base import HamiltonianModel


class KitaevModel(HamiltonianModel):
    """Kitaev Hamiltonian and time evolution. Parameters come from ``conf``."""

    name = "kitaev"
