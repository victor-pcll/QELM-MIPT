"""Kitaev model scaffold."""

from __future__ import annotations

from QELM.models.base import HamiltonianModel


class KitaevModel(HamiltonianModel):
    """Kitaev Hamiltonian and time evolution.

    Parameters belong in ``conf/models/kitaev.yaml`` and are not
    implemented. The operator of shape ``(dim, dim)`` is not built.
    """

    name = "kitaev"
