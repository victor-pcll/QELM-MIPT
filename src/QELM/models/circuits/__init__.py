"""Random brickwork circuits of two-qubit gates: U(1), Haar and Clifford.

This is the circuit model of Agrawal et al., PRX 14, 041012 (2024), used for
their charge-sharpening benchmark (issue #137). Each gate acts on two qubits
in the basis ``|00>, |01>, |10>, |11>`` as::

    u = e^{i phi_0}  (+)  U_2  (+)  e^{i phi_3}

a phase on ``|00>``, a Haar-random ``2 x 2`` unitary ``U_2`` on the
one-particle block ``{|01>, |10>}`` and a phase on ``|11>``. The gate never
changes the number of 1 bits, so the charge ``Q = sum_i b_i`` of the system
is conserved. The paper parametrizes ``U_2`` by angles ``phi_{0..3}`` and
``theta`` drawn with the measure ``sin(theta)``, which is the Haar measure
on ``U(2)``; here ``U_2`` is drawn directly from the Haar measure.

Gates act on the ``N`` system qubits, sites ``Nr .. N + Nr - 1``, only,
never on the ``Nr`` ancillas at the start of the chain. Two kinds of layers are used:

- brickwork layers: each layer is one full period of the brickwork, an even
  half-layer on the pairs ``(s_0, s_1), (s_2, s_3), ...`` followed by an odd
  half-layer on ``(s_1, s_2), (s_3, s_4), ...``, so that every bond gets a
  gate once per layer. Boundaries are open by default
  (``boundary="periodic"`` adds the pair ``(s_{N-1}, s_0)`` to the odd
  half-layer when ``N`` is even). Taking a layer as a full period is what
  reproduces the paper's Fig. 7 (issue #137);
- scrambling layers: the system qubits are shuffled and paired at random,
  all to all, which the paper applies 5 times to the product initial state
  to spread the charge before the monitored dynamics.

The same brickwork carries two other kinds of gates (issue #171), used as
the unitary dynamics of the QELM experiment in place of the TFIM:

- :class:`HaarBrickworkCircuit`: each gate is a Haar-random ``4 x 4``
  unitary (:func:`haar_unitary` with ``n = 4``), with no conserved charge;
- :class:`CliffordBrickworkCircuit`: each gate is drawn uniformly from the
  two-qubit Clifford group (:func:`random_clifford_gate`), the 11 520
  elements of the group up to a global phase. A Clifford gate maps Pauli
  strings to Pauli strings, the dynamics of the stabilizer-circuit studies
  of the measurement-induced transition.

Qubit 0 is the leftmost factor of the tensor product, as in
:func:`QELM.physics.operators.get_site_operator`.

The package is split by the nature of the gates (issue #189):

- :mod:`QELM.models.circuits.base`: :class:`BrickworkCircuit` (geometry,
  drawing and applying the layers) and the gate application;
- :mod:`QELM.models.circuits.haar`: Haar-measure gates, U(1) and U(4), and
  :class:`U1BrickworkCircuit`, :class:`HaarBrickworkCircuit`;
- :mod:`QELM.models.circuits.clifford`: the two-qubit Clifford group and
  :class:`CliffordBrickworkCircuit`;
- :mod:`QELM.models.circuits.floquet`: the deterministic Floquet Ising brick
  of Skinner, Ruhman and Nahum, PRX 9, 031009 (2019), Eq. (1), and
  :class:`FloquetBrickworkCircuit`, whose layers are half-layers, the time
  step of the paper (issue #239).

Everything is re-exported here, so ``from QELM.models.circuits import ...``
keeps working.
"""

from __future__ import annotations

from QELM.models.circuits.base import (
    BrickworkCircuit,
    Gate,
    Layer,
    apply_layer,
    apply_two_qubit_gate,
)
from QELM.models.circuits.clifford import (
    CliffordBrickworkCircuit,
    clifford_group_2q,
    random_clifford_gate,
)
from QELM.models.circuits.floquet import (
    SKINNER_2019,
    FloquetBrickworkCircuit,
    floquet_ising_gate,
)
from QELM.models.circuits.haar import (
    HaarBrickworkCircuit,
    U1BrickworkCircuit,
    haar_gate,
    haar_unitary,
    random_u1_gate,
)

__all__ = [
    "SKINNER_2019",
    "BrickworkCircuit",
    "CliffordBrickworkCircuit",
    "FloquetBrickworkCircuit",
    "Gate",
    "HaarBrickworkCircuit",
    "Layer",
    "U1BrickworkCircuit",
    "apply_layer",
    "apply_two_qubit_gate",
    "clifford_group_2q",
    "floquet_ising_gate",
    "haar_gate",
    "haar_unitary",
    "random_clifford_gate",
    "random_u1_gate",
]
