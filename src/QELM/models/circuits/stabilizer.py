"""Stabilizer simulation of the reference-qubit probe (issue #185).

The probe of Gullans & Huse, "Scalable probes of measurement-induced
criticality", PRL 125, 070606 (2020), arXiv:1910.00020, on the Clifford
brickwork of :class:`~QELM.models.circuits.clifford.CliffordBrickworkCircuit`:

1. the reference qubit ``R`` (qubit 0) starts in a Bell pair with one system
   site (:class:`QELM.encoding.AncillaBellPairEncoding`), the other ``N``
   system qubits (qubits ``1 .. N``) in ``|0>``;
2. ``scramble_steps`` steps without measurement (``t0 = 2N`` in the paper)
   turn the system into a pseudo-random state;
3. ``steps`` more steps (``2N`` in the paper) in which every system qubit is
   measured in ``Z`` with probability ``p`` after the gates;
4. the order parameter is the entropy ``S_Q`` of ``R``, averaged over
   circuits and trajectories.

A step is one half-layer of the brickwork with periodic boundaries: the
even bonds ``(0, 1), (2, 3), ...`` at even steps, the odd bonds
``(1, 2), ..., (N - 1, 0)`` at odd steps. Each site gets one gate per
step, so the step is the paper's time unit, "the number of two-qubit gates
that have acted on each site". Each gate is drawn uniformly from the
11 520 elements of :func:`~QELM.models.circuits.clifford.clifford_group_2q`,
anew for every trajectory: the paper averages over circuits.

The state stays a stabilizer state, so it is simulated with the tableau
simulator of ``stim`` in polynomial time, which reaches the paper's sizes
``N = 64`` to ``256``. For a stabilizer state the reduced state of ``R`` is
either pure or ``I / 2``, so ``S_Q`` is 0 or 1 bit in every trajectory, and
it does not depend on the measurement outcomes, only on which sites were
measured. :func:`draw_probe_history` draws the gates and the measured sites
with NumPy, so a run is reproducible from its seed, and
:func:`reference_entropy_history` replays them; the tests replay the same
history on the state vector, with the gates of the library, and find the
same ``S_Q`` at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import stim  # type: ignore[import-untyped]

from QELM.models.circuits.clifford import clifford_group_2q

# The reference qubit R is qubit 0; system site k is qubit k + 1.
REFERENCE = 0


@cache
def clifford_gate_programs() -> tuple[str, ...]:
    """Return each two-qubit Clifford gate as a ``stim`` program template.

    Element ``k`` is :func:`clifford_group_2q` ``()[k]``, converted with
    ``stim.Tableau.from_unitary_matrix`` (big endian: the first target is
    the most significant bit of the gate index, as in
    :func:`~QELM.models.circuits.base.apply_two_qubit_gate`) and decomposed
    into ``H``, ``S`` and ``CX`` gates. The two qubits are the fields
    ``{a}`` and ``{b}``. Built once and cached.

    Returns:
        ``11 520`` program templates, in the order of the group.
    """
    programs = []
    for u in clifford_group_2q():
        circuit = stim.Tableau.from_unitary_matrix(u, endian="big").to_circuit(
            method="elimination"
        )
        lines = []
        for instruction in circuit:
            fields = " ".join(
                "{a}" if t.value == 0 else "{b}" for t in instruction.targets_copy()
            )
            lines.append(f"{instruction.name} {fields}")
        programs.append("\n".join(lines))
    return tuple(programs)


def periodic_bonds(N: int, half: int) -> list[tuple[int, int]]:
    """Return the bonds of one half-layer of the periodic brickwork.

    Args:
        N: Number of system sites, even.
        half: 0 for the even bonds ``(0, 1), (2, 3), ...``, 1 for the odd
            bonds ``(1, 2), ..., (N - 1, 0)``; only the parity counts.

    Returns:
        ``N // 2`` disjoint pairs of system sites (from 0).

    Raises:
        ValueError: If ``N`` is odd or below 2.
    """
    if N < 2 or N % 2:
        raise ValueError(f"The periodic brickwork needs an even N >= 2, got {N}.")
    start = half % 2
    return [(k, (k + 1) % N) for k in range(start, N, 2)]


@dataclass(frozen=True)
class ProbeHistory:
    """The random draws of one trajectory of the probe.

    Attributes:
        gates: Index in :func:`clifford_group_2q` of each gate, shape
            ``(T, N // 2)``, in the order of :func:`periodic_bonds` at each
            step.
        measured: ``True`` where a system site is measured after the gates
            of a step, shape ``(T, N)``. The ``scramble_steps`` first rows
            are ``False``.
    """

    gates: np.ndarray
    measured: np.ndarray

    @property
    def N(self) -> int:
        """Number of system sites."""
        return int(self.measured.shape[1])

    @property
    def T(self) -> int:
        """Number of steps, scrambling included."""
        return int(self.measured.shape[0])


def draw_probe_history(
    N: int, p: float, scramble_steps: int, steps: int, rng: np.random.Generator
) -> ProbeHistory:
    """Draw the gates and the measured sites of one trajectory.

    Args:
        N: Number of system sites, even.
        p: Probability to measure each site after each measured step.
        scramble_steps: Steps without measurement, at the start.
        steps: Measured steps, after the scrambling.
        rng: Random generator.

    Returns:
        The history of ``scramble_steps + steps`` steps.

    Raises:
        ValueError: If ``N`` is odd, ``p`` is not a probability or a number
            of steps is negative.
    """
    periodic_bonds(N, 0)  # checks N
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    if scramble_steps < 0 or steps < 0:
        raise ValueError(
            f"Numbers of steps must be >= 0, got {scramble_steps}, {steps}."
        )
    T = scramble_steps + steps
    gates = rng.integers(len(clifford_gate_programs()), size=(T, N // 2))
    measured = rng.random((T, N)) < p
    measured[:scramble_steps] = False
    return ProbeHistory(gates=gates, measured=measured)


def reference_entropy_history(
    history: ProbeHistory, site: int | None = None, seed: int = 0
) -> np.ndarray:
    """Replay a history on the stabilizer state and return ``S_Q`` after each step.

    Args:
        history: Output of :func:`draw_probe_history`.
        site: System site paired with ``R`` at ``t = 0``; ``None`` for
            ``N // 2``.
        seed: Seed of the ``stim`` outcomes. ``S_Q`` does not depend on it.

    Returns:
        Entropy of ``R`` in bits, 0 or 1, after the gates and measurements
        of each step, shape ``(T,)``.

    Raises:
        ValueError: If ``site`` is not a system site.
    """
    N = history.N
    site = N // 2 if site is None else int(site)
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    programs = clifford_gate_programs()
    bonds = [periodic_bonds(N, 0), periodic_bonds(N, 1)]
    sim = stim.TableauSimulator(seed=seed)
    # The encoding: (|00> + |11>) / sqrt(2) on R and the site.
    sim.h(REFERENCE)
    sim.cnot(REFERENCE, site + 1)
    identity = stim.PauliString(1)
    S = np.zeros(history.T)
    for t in range(history.T):
        lines = [
            programs[g].format(a=a + 1, b=b + 1)
            for g, (a, b) in zip(history.gates[t], bonds[t % 2], strict=True)
        ]
        sites = np.flatnonzero(history.measured[t])
        if sites.size:
            lines.append("M " + " ".join(str(k + 1) for k in sites))
        sim.do_circuit(stim.Circuit("\n".join(lines)))
        # R is mixed iff no stabilizer acts on it alone: its Bloch vector is I.
        S[t] = 1.0 if sim.peek_bloch(REFERENCE) == identity else 0.0
    return S
