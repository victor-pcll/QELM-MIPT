"""Reference qubit of the purification-decoding task (issue #247).

The probe of Gullans & Huse (arXiv:1905.05195, arXiv:1910.00020): a
reference qubit ``R`` is maximally entangled with one qubit of the system at
``t = 0`` and never touched again; the system alone evolves under unitary
steps and local measurements. ``R`` stays mixed as long as the system hides
its information (volume-law, mixed phase) and becomes pure once the
measurements have revealed it (area-law, pure phase).

The state is a vector of ``N + 1`` qubits with ``R`` first (qubit 0, the
ancilla position of the project, ``Nr = 1``) and the ``N`` system qubits
after it, qubit 0 being the leftmost factor of the tensor product as in
:func:`QELM.physics.operators.get_site_operator`. Reshaped to ``(2, 2**N)``,
row ``r`` holds the system part paired with ``|r>_R``, so a system unitary
``U`` acts as ``psi.reshape(2, -1) @ U.T`` and the reduced state of ``R`` is
``A @ A^dagger`` with ``A = psi.reshape(2, -1)``.

Measuring ``R`` in Z at the end has the Born probability ``(1 + z) / 2`` of
``+1``, with ``z = <Z_R>`` in the state conditioned on the record. Knowing
the full state, the best guess of that outcome is ``sign(z)``, right with
probability ``(1 + |z|) / 2``: :func:`bayes_accuracy` is this ceiling of any
decoder of the record, and :func:`expected_accuracy` scores a decoder's
guesses against ``z`` without the noise of the single shots.
"""

from __future__ import annotations

import numpy as np


def _system_qubits(psi: np.ndarray) -> int:
    dim = psi.shape[0]
    n = dim.bit_length() - 1
    if psi.ndim != 1 or dim < 4 or 2**n != dim:
        raise ValueError(
            f"psi must be a state vector of R and N >= 1 system qubits, shape "
            f"(2**(N + 1),), got {psi.shape}."
        )
    return n - 1


def reference_bell_state(system: np.ndarray, site: int) -> np.ndarray:
    """Entangle a reference qubit maximally with qubit ``site`` of a product state.

    Returns ``(|0>_R |b(0)> + |1>_R |b(1)>) / sqrt(2)``, where ``|b(r)>`` is
    the computational basis state ``system`` with qubit ``site`` set to
    ``r``: a Bell pair between ``R`` and ``site``, the other system qubits
    left as they are.

    Args:
        system: Computational basis state of the ``N`` system qubits, a
            vector of shape ``(2**N,)`` with a single nonzero entry (for
            instance the Néel state).
        site: System qubit paired with ``R``, from 0 to ``N - 1``.

    Returns:
        State vector of shape ``(2**(N + 1),)``, ``R`` first.

    Raises:
        ValueError: If ``system`` is not a basis state of qubits or ``site``
            is out of range.
    """
    system = np.asarray(system)
    dim = system.shape[0]
    N = dim.bit_length() - 1
    nonzero = np.flatnonzero(system)
    if system.ndim != 1 or 2**N != dim or N < 1 or nonzero.size != 1:
        raise ValueError(
            "system must be a computational basis state of shape (2**N,), N >= 1."
        )
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    mask = 1 << (N - 1 - site)
    psi = np.zeros((2, dim), dtype=complex)
    psi[0, int(nonzero[0]) & ~mask] = 1.0
    psi[1, int(nonzero[0]) | mask] = 1.0
    return psi.reshape(-1) / np.sqrt(2.0)


def reference_rho(psi: np.ndarray) -> np.ndarray:
    """Return the ``2 x 2`` reduced density matrix of ``R`` (qubit 0) of ``psi``."""
    N = _system_qubits(psi)
    A = np.asarray(psi, dtype=complex).reshape(2, 2**N)
    return np.asarray(A @ A.conj().T)


def reference_bloch(psi: np.ndarray) -> np.ndarray:
    """Return the Bloch vector ``(<X_R>, <Y_R>, <Z_R>)`` of the reference qubit.

    With ``rho_R = (I + x X + y Y + z Z) / 2``: ``x = 2 Re rho_01``,
    ``y = -2 Im rho_01`` and ``z = rho_00 - rho_11``. Its length is 1 for a
    pure ``R`` (purified) and 0 for a maximally mixed one.
    """
    rho = reference_rho(psi)
    return np.array(
        [
            2.0 * rho[0, 1].real,
            -2.0 * rho[0, 1].imag,
            (rho[0, 0] - rho[1, 1]).real,
        ]
    )


def reference_entropy(bloch: np.ndarray) -> float:
    """Return the von Neumann entropy of ``R``, in bits, from its Bloch vector.

    The eigenvalues of ``rho_R`` are ``(1 +- |r|) / 2``: 1 bit for a
    maximally mixed ``R``, 0 for a purified one.
    """
    r = min(float(np.linalg.norm(bloch)), 1.0)
    values = np.array([(1.0 + r) / 2.0, (1.0 - r) / 2.0])
    values = values[values > 1e-15]
    return float(-np.sum(values * np.log2(values)))


def bayes_accuracy(z: np.ndarray) -> float:
    """Best mean accuracy of a guess of the Z outcome of ``R``, ``mean((1 + |z|) / 2)``.

    Any decoder of the record, however nonlinear, sees at most the state
    conditioned on the record, where the outcome is +1 with probability
    ``(1 + z) / 2``: guessing ``sign(z)`` is optimal. 1/2 when ``R`` is
    mixed (``z = 0``), 1 when every ``R`` is purified along Z.
    """
    z = np.asarray(z, dtype=float)
    return float(np.mean((1.0 + np.abs(z)) / 2.0))


def expected_accuracy(z: np.ndarray, guess: np.ndarray) -> float:
    """Mean probability that the guesses ``sign(guess)`` match the Z outcome of ``R``.

    A guess ``s = +-1`` (``guess >= 0`` counts as +1) is right with
    probability ``(1 + s z) / 2``. Averaging this probability instead of
    scoring single shots removes their noise; it never exceeds
    :func:`bayes_accuracy`.

    Raises:
        ValueError: If the shapes differ.
    """
    z = np.asarray(z, dtype=float)
    guess = np.asarray(guess, dtype=float)
    if z.shape != guess.shape:
        raise ValueError(f"z and guess differ in shape: {z.shape}, {guess.shape}.")
    sign = np.where(guess >= 0.0, 1.0, -1.0)
    return float(np.mean((1.0 + sign * z) / 2.0))


def entangle_reference(
    psi: np.ndarray, site: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Entangle ``R`` maximally with system qubit ``site`` of a state where ``R = |0>``.

    The protocol of Gullans & Huse, after a scrambling phase: the system is
    first evolved alone, then ``R`` is paired with one of its qubits. Here
    qubit ``site`` is measured in Z (outcome ``m`` with the Born rule), which
    leaves the qubit in ``|m>``; it is reset to ``|0>`` (``X`` when ``m =
    1``), giving ``|psi_0>``, and the state becomes ``(|0>|psi_0> + |1>
    X_site |psi_0>) / sqrt(2)``: the Bell pair ``|00> + |11>`` between ``R``
    and the qubit, whatever ``m``, the other system qubits keeping the
    entanglement built so far. ``S_R = 1`` bit exactly. The reset keeps the
    sign of ``<Z_R>`` tied to that of the qubit, as without scrambling; a
    pair that depends on ``m`` would flip the map from the record to ``R``
    with an outcome the decoder does not see.

    Args:
        psi: State vector of ``R`` (qubit 0, in ``|0>``) and ``N`` system
            qubits, shape ``(2**(N + 1),)``.
        site: System qubit paired with ``R``, from 0 to ``N - 1``.
        rng: Generator of the outcome of the measurement.

    Returns:
        ``(psi, m)``: the new state and the outcome, +-1, of the measurement
        of ``site``.

    Raises:
        ValueError: If ``R`` is not in ``|0>`` or ``site`` is out of range.
    """
    N = _system_qubits(psi)
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    A = np.asarray(psi, dtype=complex).reshape(2, 2**N)
    if np.linalg.norm(A[1]) > 1e-9:
        raise ValueError("R must be in |0> before it is entangled.")
    bits = (np.arange(2**N) >> (N - 1 - site)) & 1
    p0 = float(np.sum(np.abs(A[0, bits == 0]) ** 2))
    outcome = 0 if rng.random() < p0 else 1
    kept = np.where(bits == outcome, A[0], 0.0)
    kept = kept / np.linalg.norm(kept)
    flip = np.arange(2**N) ^ (1 << (N - 1 - site))
    if outcome == 1:
        # Reset the qubit to |0>, so that the pair is always |00> + |11>.
        kept = kept[flip]
    out = np.stack([kept, kept[flip]]) / np.sqrt(2.0)
    return out.reshape(-1), 1 - 2 * outcome
