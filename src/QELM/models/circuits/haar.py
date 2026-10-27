"""Haar-measure gates and circuits: U(1) and U(4) brickworks (issue #189).

- :func:`haar_unitary`: an ``n x n`` unitary from the Haar measure;
- :func:`random_u1_gate` and :class:`U1BrickworkCircuit`: charge-conserving
  gates, a phase on ``|00>`` and ``|11>`` and a Haar ``2 x 2`` block on
  ``{|01>, |10>}`` (Agrawal et al., PRX 14, 041012 (2024), issue #137);
- :func:`haar_gate` and :class:`HaarBrickworkCircuit`: Haar-random
  ``4 x 4`` gates, no conserved charge (issue #171).
"""

from __future__ import annotations

import numpy as np

from QELM.models.circuits.base import BrickworkCircuit, _generator


def haar_unitary(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw an ``n x n`` unitary from the Haar measure.

    QR decomposition of a complex Gaussian matrix, with the phases of the
    diagonal of ``R`` moved into ``Q`` so that the distribution is exactly
    Haar (Mezzadri, Notices AMS 54, 592 (2007)).

    Args:
        n: Matrix size, at least 1.
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex unitary matrix of shape ``(n, n)``.
    """
    gen = _generator(rng)
    z = (gen.standard_normal((n, n)) + 1j * gen.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return np.asarray(q * (d / np.abs(d)))


def random_u1_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a random charge-conserving two-qubit gate.

    Args:
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex ``4 x 4`` unitary in the basis ``|00>, |01>, |10>, |11>``:
        uniform phases on ``|00>`` and ``|11>`` and a Haar ``2 x 2`` block
        on ``{|01>, |10>}``.
    """
    gen = _generator(rng)
    phi_0, phi_3 = gen.uniform(0.0, 2 * np.pi, size=2)
    u = np.zeros((4, 4), dtype=complex)
    u[0, 0] = np.exp(1j * phi_0)
    u[1:3, 1:3] = haar_unitary(2, gen)
    u[3, 3] = np.exp(1j * phi_3)
    return u


def haar_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a Haar-random two-qubit gate, a ``4 x 4`` unitary of U(4).

    Args:
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex ``4 x 4`` unitary in the basis ``|00>, |01>, |10>, |11>``.
    """
    return haar_unitary(4, rng)


class U1BrickworkCircuit(BrickworkCircuit):
    """Brickwork circuit of random U(1) gates on the system qubits.

    Same geometry and parameters as :class:`BrickworkCircuit`, with each
    gate drawn by :func:`random_u1_gate`: the charge of the system is
    conserved. The circuit of Agrawal et al. (2024), issue #137. Config
    block ``U1_circuit``.
    """

    name: str = "U1_circuit"

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a random U(1) gate."""
        return random_u1_gate(rng)


class HaarBrickworkCircuit(BrickworkCircuit):
    """Brickwork circuit of Haar-random two-qubit gates on the system qubits.

    Same geometry and parameters as :class:`BrickworkCircuit`, with each
    gate drawn from the Haar measure on U(4) (:func:`haar_gate`): no charge
    is conserved. Config block ``haar_circuit`` (issue #171).
    """

    name: str = "haar_circuit"

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a Haar-random ``4 x 4`` unitary."""
        return haar_gate(rng)
