"""Quantum primitives: Pauli matrices and their embedding in an N-qubit chain.

Measurements and feature extraction live in :mod:`QELM.training`.
"""

from __future__ import annotations

from QELM.physics.operators import (
    PAULIS,
    I,
    X,
    Y,
    Z,
    get_site_operator,
    local_expectation,
    pauli,
    qubit_counts,
)

__all__ = [
    "PAULIS",
    "I",
    "X",
    "Y",
    "Z",
    "get_site_operator",
    "local_expectation",
    "qubit_counts",
    "pauli",
]
