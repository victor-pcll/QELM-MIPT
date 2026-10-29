"""Local Floquet Ising chain of Skinner, Ruhman and Nahum (issue #239).

The nearest-neighbour model of the numerics of B. Skinner, J. Ruhman and
A. Nahum, PRX 9, 031009 (2019), their Eq. (1). Every brick of the running-bond
circuit is the same two-qubit unitary on the sites ``x, x + 1``::

    U_{x,x+1} = exp{-i [phi_xx X_x X_{x+1} + theta_x (X_x + X_{x+1})]}
              x exp{-i [phi_zz Z_x Z_{x+1} + theta_z (Z_x + Z_{x+1})]}

with Pauli matrices (the right factor acts first), and the paper's values
``phi_xx = 0.3``, ``phi_zz = 0.6``, ``theta_x = 0.2``, ``theta_z = 0.4``. It is
an Ising chain in a tilted field: the couplings ``Z Z`` and the longitudinal
field ``theta_z Z`` on one side, the transverse field ``theta_x X`` (and a
coupling ``X X``) on the other, which do not commute, so the dynamics is
chaotic. Nothing is conserved: neither the charge ``sum_i Z_i`` nor its parity.

The paper's time step is one layer of bricks, on the even bonds or on the odd
ones in turn (two layers make one period), and each qubit is measured in Z
with probability ``p`` after each layer. :class:`FloquetBrickworkCircuit`
follows it: its layer ``t`` is a *half*-layer of the brickwork of
:class:`~QELM.models.circuits.base.BrickworkCircuit`, the even bonds for even
``t`` and the odd bonds for odd ``t``, so that a measurement round after each
layer is the protocol of the paper.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.linalg

from QELM.models.circuits.base import BrickworkCircuit

# Eq. (1) of Skinner, Ruhman and Nahum, PRX 9, 031009 (2019), Sec. II.
SKINNER_2019 = {"phi_xx": 0.3, "phi_zz": 0.6, "theta_x": 0.2, "theta_z": 0.4}

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def floquet_ising_gate(
    phi_xx: float, phi_zz: float, theta_x: float, theta_z: float
) -> np.ndarray:
    """Return the brick ``U_{x,x+1}`` of Eq. (1) of Skinner et al. (2019).

    Args:
        phi_xx: Coupling of ``X_x X_{x+1}``.
        phi_zz: Coupling of ``Z_x Z_{x+1}``.
        theta_x: Transverse field, on both sites.
        theta_z: Longitudinal field, on both sites.

    Returns:
        Complex ``4 x 4`` unitary in the basis ``|00>, |01>, |10>, |11>``,
        site ``x`` the most significant bit, ``|0>`` the ``Z = +1`` state.
    """
    xx = np.kron(_X, _X)
    zz = np.kron(_Z, _Z)
    x_fields = np.kron(_X, _I2) + np.kron(_I2, _X)
    z_fields = np.kron(_Z, _I2) + np.kron(_I2, _Z)
    u_x = scipy.linalg.expm(-1j * (phi_xx * xx + theta_x * x_fields))
    u_z = scipy.linalg.expm(-1j * (phi_zz * zz + theta_z * z_fields))
    return np.asarray(u_x @ u_z)


class FloquetBrickworkCircuit(BrickworkCircuit):
    """Running-bond circuit of the Floquet Ising brick of Skinner et al. (2019).

    The geometry of :class:`BrickworkCircuit` (system qubits ``Nr .. N-1``,
    open or periodic boundaries), with two differences:

    - every gate is the same deterministic brick :func:`floquet_ising_gate`,
      with ``phi_xx``, ``phi_zz``, ``theta_x`` and ``theta_z`` from the config
      (defaults: the paper's values, :data:`SKINNER_2019`). The circuit has no
      randomness, so :meth:`redraw` gives the same gates;
    - layer ``t`` is one half-layer, the even bonds for even ``t`` and the odd
      bonds for odd ``t``: the time step of the paper, after which every qubit
      can be measured. ``depth`` counts these half-layers, so ``depth = 2``
      is one Floquet period.

    Config block ``floquet_circuit`` (``conf/models/floquet_circuit.yaml``).
    """

    name: str = "floquet_circuit"
    layers_per_period: int = 2

    def __init__(
        self, params: Mapping[str, Any], rng: np.random.Generator | None = None
    ) -> None:
        """Store the couplings, then the geometry (see :class:`BrickworkCircuit`).

        Args:
            params: As for :class:`BrickworkCircuit`, plus the optional
                ``phi_xx``, ``phi_zz``, ``theta_x`` and ``theta_z``.
            rng: Unused, the circuit is deterministic; kept for the common
                signature.

        Raises:
            ValueError: If ``scrambling_layers`` is set (the paper has none),
                or as :class:`BrickworkCircuit`.
        """
        if int(params.get("scrambling_layers", 0)) != 0:
            raise ValueError("floquet_circuit has no scrambling layers.")
        couplings = {
            key: float(params.get(key, value)) for key, value in SKINNER_2019.items()
        }
        self.phi_xx = couplings["phi_xx"]
        self.phi_zz = couplings["phi_zz"]
        self.theta_x = couplings["theta_x"]
        self.theta_z = couplings["theta_z"]
        self.gate = floquet_ising_gate(**couplings)
        super().__init__(params, rng)

    def redraw(self, rng: np.random.Generator | None = None) -> None:
        """Build the ``depth`` half-layers, even bonds first.

        Args:
            rng: Unused, the gates are fixed.
        """
        self.scrambling = []
        self.layers = [
            [(i, j, self.gate) for i, j in self.bonds(t)] for t in range(self.depth)
        ]

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Return the brick of Eq. (1); ``rng`` is not used."""
        return self.gate
