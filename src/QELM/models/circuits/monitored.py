"""State-vector simulation of the monitored Haar brickwork (issue #257).

The same circuit, step and boundaries as the stabilizer probes of
:mod:`QELM.models.circuits.stabilizer`, with Haar-random ``U(4)`` gates in
place of the Clifford ones: one step is one half-layer of the periodic
brickwork, then a Z measurement of every system site with probability
``p``, outcomes drawn with the Born rule; new gates for every trajectory.
The state is no longer a stabilizer state, so it is a state vector, and the
von Neumann and Renyi-2 entropies differ.

Two probes, as for the Clifford circuit:

- :func:`haar_reference_entropy`: the reference qubit of Gullans & Huse
  (2020), in a Bell pair with the middle site at ``t = 0``, then
  ``scramble_steps`` steps without measurement and ``steps`` measured ones;
- :func:`haar_steady_state_entropies`: the half-chain entropy and the
  tripartite mutual information of a chain monitored from ``|0...0>``.
"""

from __future__ import annotations

import numpy as np

from QELM.encoding.bell_pair import AncillaBellPairEncoding
from QELM.evaluation.entropy import renyi_entropy, von_neumann_entropy
from QELM.models.circuits.base import apply_two_qubit_gate
from QELM.models.circuits.haar import haar_unitary
from QELM.models.circuits.stabilizer import periodic_bonds, quarters


def measure_z(
    psi: np.ndarray, qubit: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Measure ``Z`` on one qubit of a state vector, with the Born rule.

    Args:
        psi: Normalized state of ``n`` qubits, qubit 0 the most significant.
        qubit: Measured qubit.
        rng: Random generator of the outcome.

    Returns:
        The normalized post-measurement state and the outcome, 0 or 1.
    """
    n = psi.size.bit_length() - 1
    T = psi.reshape(2**qubit, 2, 2 ** (n - qubit - 1))
    p1 = float(np.sum(np.abs(T[:, 1]) ** 2))
    outcome = int(rng.random() < p1)
    out = np.zeros_like(T)
    out[:, outcome] = T[:, outcome] / np.sqrt(p1 if outcome else 1.0 - p1)
    return out.reshape(-1), outcome


def _evolve(
    psi: np.ndarray,
    N: int,
    offset: int,
    p_of_step: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one Haar half-layer, then the measurements, per entry of ``p_of_step``."""
    for t, p in enumerate(p_of_step):
        for a, b in periodic_bonds(N, t):
            psi = apply_two_qubit_gate(
                psi, haar_unitary(4, rng), a + offset, b + offset
            )
        for k in np.flatnonzero(rng.random(N) < p):
            psi, _ = measure_z(psi, int(k) + offset, rng)
    return psi


def haar_reference_entropy(
    N: int, p: float, scramble_steps: int, steps: int, rng: np.random.Generator
) -> dict[str, float]:
    """Entropy of the reference qubit after one trajectory of the Haar probe.

    Args:
        N: Number of system sites, even.
        p: Measurement probability per site and measured step.
        scramble_steps: Steps without measurement after the encoding.
        steps: Measured steps.
        rng: Random generator of the gates, sites and outcomes.

    Returns:
        ``{"S_Q": von Neumann entropy, "S_Q2": Renyi-2 entropy}`` of ``R``,
        in bits, at the final step.
    """
    psi = AncillaBellPairEncoding({"N": N}).state()
    psi = _evolve(psi, N, 1, [0.0] * scramble_steps + [p] * steps, rng)
    return {
        "S_Q": von_neumann_entropy(psi, [0]),
        "S_Q2": renyi_entropy(psi, 2, [0]),
    }


def haar_steady_state_entropies(
    N: int, p: float, steps: int, rng: np.random.Generator
) -> dict[str, float]:
    """Half-chain entropies and ``I_3`` after ``steps`` monitored steps.

    The chain starts in ``|0...0>`` and is measured from the first step;
    ``A, B, C, D`` are the quarters of
    :func:`~QELM.models.circuits.stabilizer.quarters`.

    Args:
        N: Number of system sites, divisible by 4.
        p: Measurement probability per site and step.
        steps: Number of steps.
        rng: Random generator of the gates, sites and outcomes.

    Returns:
        ``S_half`` and ``S_half2`` (von Neumann and Renyi-2 entropies of
        ``A B``) and ``I3`` and ``I3_2`` (tripartite mutual information of
        ``A, B, C`` with each entropy), in bits.
    """
    A, B, C, _ = quarters(N)
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0
    psi = _evolve(psi, N, 0, [p] * steps, rng)
    regions = {
        "A": A,
        "B": B,
        "C": C,
        "AB": A + B,
        "BC": B + C,
        "AC": A + C,
        "ABC": A + B + C,
    }
    out: dict[str, float] = {}
    for suffix, entropy in (
        ("", von_neumann_entropy),
        ("2", lambda s, r: renyi_entropy(s, 2, r)),
    ):
        S = {name: entropy(psi, region) for name, region in regions.items()}
        I3 = S["A"] + S["B"] + S["C"] - S["AB"] - S["BC"] - S["AC"] + S["ABC"]
        out[f"S_half{suffix}"] = S["AB"]
        out["I3" + ("_2" if suffix else "")] = I3
    return out
