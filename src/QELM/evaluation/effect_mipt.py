"""Reference MIPT probes and Born-weighted frame dimension, convention of #314.

The effect-space rank of :mod:`QELM.evaluation.effect_rank` counts every
direction of the span of the effects, whatever its weight. This module gives
the quantities that are needed to compare it with the measurement-induced
transition on the **same circuits** (issue #322):

- the reference probes of the transition, in the layer convention of
  :mod:`QELM.evaluation.effect_rank` (open brickwork, one layer = even bonds
  then odd bonds, then the ``Z`` measurements of the sites of the mask):

  - :func:`clifford_purification`: entropy of an initially maximally mixed
    system (Gullans & Huse, PRX 10, 041020 (2020)), and of a reference qubit
    entangled with the middle site after a scrambling stage, for Clifford
    circuits, with ``stim``. For a stabilizer state both depend only on the
    gates and on the measured sites, not on the outcomes;
  - :func:`haar_reference_entropy`: entropy of the reference qubit for Haar
    circuits, state vector, outcomes drawn with the Born rule;

- the Born-weighted counterpart of the rank, :func:`frame_potential`.
  With the final readout every effect has rank one, ``E_r = |phi_r><phi_r|``
  with ``phi_r = K_m^dagger |z>``. Drawing records ``r = (m, z)`` with the
  Born rule for the maximally mixed input, the second frame potential of the
  normalised effects is ``Phi_2 = E_{r, r'} |<phi_r|phi_r'>|^4`` (hats
  omitted), and ``D_2 = 1 / Phi_2`` is the effective dimension: ``2^N`` for a
  projective measurement in one basis, about ``4^N / 2`` for a Haar POVM. The
  rank is the number of non-zero eigenvalues of
  ``F = E_r |e_r)(e_r|``; ``Phi_2 = Tr F^2``.

  A record is drawn by running a Haar-random input state forward with the
  Born rule (the average of ``|psi><psi|`` is ``I / 2^N``), and ``phi_r`` is
  then obtained by running ``|z>`` backwards through the same gates and
  projectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import stim  # type: ignore[import-untyped]

from QELM.evaluation.effect_rank import brick_pairs
from QELM.models.circuits.clifford import clifford_group_2q
from QELM.models.circuits.haar import haar_gate
from QELM.models.circuits.stabilizer import (
    clifford_gate_programs,
    stabilizer_entropy,
    stabilizer_matrix,
)

# ---------------------------------------------------------------------------
# State-vector helpers (qubit 0 is the most significant bit)
# ---------------------------------------------------------------------------


def apply_gate(psi: np.ndarray, n: int, site: int, gate: np.ndarray) -> np.ndarray:
    """Apply a two-qubit gate on the sites ``(site, site + 1)`` of ``psi``.

    Args:
        psi: State vectors, shape ``(..., 2**n)``; leading axes are batched.
        n: Number of qubits.
        site: Left site of the gate.
        gate: ``4 x 4`` unitary in the basis ``|q_site q_{site+1}>``.

    Returns:
        The new state vectors, same shape.
    """
    lead = psi.shape[:-1]
    v = psi.reshape(*lead, 2**site, 4, 2 ** (n - site - 2))
    out = np.einsum("ab,...lbr->...lar", gate, v)
    return np.asarray(out).reshape(psi.shape)


def project(psi: np.ndarray, n: int, site: int, outcome: int) -> np.ndarray:
    """Return ``P_outcome`` of ``site`` applied to ``psi`` (not normalised)."""
    lead = psi.shape[:-1]
    v = psi.reshape(*lead, 2**site, 2, 2 ** (n - site - 1)).copy()
    v[..., 1 - outcome, :] = 0.0
    return v.reshape(psi.shape)


def prob_zero(psi: np.ndarray, n: int, site: int) -> float:
    """Born probability of the outcome 0 of a ``Z`` measurement of ``site``."""
    v = psi.reshape(2**site, 2, 2 ** (n - site - 1))
    return float(np.sum(np.abs(v[:, 0, :]) ** 2) / np.sum(np.abs(psi) ** 2))


def haar_state(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure state on ``n`` qubits."""
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return np.asarray(psi / np.linalg.norm(psi))


def draw_haar_circuit(
    n: int, depth: int, rng: np.random.Generator
) -> list[list[np.ndarray]]:
    """Haar gates ``gates[t][g]`` of an open brickwork, order of :func:`brick_pairs`."""
    return [[haar_gate(rng) for _ in brick_pairs(n)] for _ in range(depth)]


# ---------------------------------------------------------------------------
# Born-weighted frame potential
# ---------------------------------------------------------------------------


def sample_record_vector(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Draw one record with the Born rule and return its normalised ``phi``.

    The input is a Haar-random state, so the record ``r = (m, z)`` has the
    probability ``Tr[E_r] / 2^N``, that of the maximally mixed input.

    Args:
        gates: ``gates[t][g]``, the two-qubit gates of layer ``t``.
        masks: Boolean ``(T, N)``, the mid-circuit measurements.
        rng: Random generator.

    Returns:
        ``(phi, k)``: the unit vector ``K_m^dagger |z> / ||.||`` and the
        number ``k`` of mid-circuit outcomes of the record.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    pairs = brick_pairs(n)
    psi = haar_state(n, rng)
    outcomes: list[list[tuple[int, int]]] = []
    for t in range(depth):
        for site, gate in zip(pairs, gates[t], strict=True):
            psi = apply_gate(psi, n, site, gate)
        layer = []
        for site in np.flatnonzero(masks[t]).tolist():
            p0 = prob_zero(psi, n, int(site))
            o = 0 if rng.random() < p0 else 1
            psi = project(psi, n, int(site), o)
            psi /= np.linalg.norm(psi)
            layer.append((int(site), o))
        outcomes.append(layer)
    prob = np.abs(psi) ** 2
    z = int(rng.choice(2**n, p=prob / prob.sum()))
    phi = np.zeros(2**n, dtype=complex)
    phi[z] = 1.0
    for t in reversed(range(depth)):
        for site, o in outcomes[t]:
            phi = project(phi, n, site, o)
        for site, gate in zip(reversed(pairs), reversed(list(gates[t])), strict=True):
            phi = apply_gate(phi, n, site, gate.conj().T)
    norm = np.linalg.norm(phi)
    return phi / norm, sum(len(layer) for layer in outcomes)


@dataclass(frozen=True)
class FramePotential:
    """Estimate of the second frame potential of the normalised effects.

    Attributes:
        phi2: ``E |<phi_r|phi_r'>|^4`` over pairs of distinct samples.
        sem: Standard error of ``phi2`` (from the per-sample means).
        log2_d2: ``-log2(phi2)``, the effective dimension in bits.
        samples: Number of records drawn.
    """

    phi2: float
    sem: float
    log2_d2: float
    samples: int


def frame_potential(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> FramePotential:
    """Estimate ``Phi_2 = Tr F^2`` from ``samples`` Born-drawn records.

    ``Phi_2`` is the mean of ``|<phi_a|phi_b>|^4`` over all pairs ``a != b``,
    an unbiased U-statistic. Its error is estimated from the spread of the
    per-sample means, with the factor 2 of a U-statistic of order two.
    """
    phis = np.array(
        [sample_record_vector(gates, masks, rng)[0] for _ in range(samples)]
    )
    g = np.abs(phis.conj() @ phis.T) ** 4
    np.fill_diagonal(g, 0.0)
    per = g.sum(axis=1) / (samples - 1)
    phi2 = float(per.mean())
    sem = float(2 * per.std(ddof=1) / np.sqrt(samples))
    return FramePotential(
        phi2=phi2, sem=sem, log2_d2=float(-np.log2(phi2)), samples=samples
    )


def frame_potential_exact(
    gates: Sequence[Sequence[np.ndarray]], masks: np.ndarray
) -> float:
    """``Phi_2`` by enumeration of every record, for small ``N`` (tests).

    Each Kraus operator ``K_r = <z| ... P U ...`` is built explicitly;
    ``p_r = Tr[K_r^dagger K_r] / 2^N`` and ``Phi_2 = sum p_r p_r' |<phi_r|phi_r'>|^4``.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    pairs = brick_pairs(n)
    dim = 2**n
    branches: list[np.ndarray] = [np.eye(dim, dtype=complex)]
    for t in range(depth):
        new = []
        for k in branches:
            for site, gate in zip(pairs, gates[t], strict=True):
                k = apply_gate(k.T, n, site, gate).T
            new.append(k)
        branches = new
        for site in np.flatnonzero(masks[t]).tolist():
            branches = [
                project(k.T, n, int(site), o).T for k in branches for o in (0, 1)
            ]
    vecs, weights = [], []
    for k in branches:
        for z in range(dim):
            phi = k[z].conj()
            w = float(np.sum(np.abs(phi) ** 2))
            if w > 1e-14:
                vecs.append(phi / np.sqrt(w))
                weights.append(w / dim)
    v = np.array(vecs)
    wts = np.array(weights)
    g = np.abs(v.conj() @ v.T) ** 4
    return float(wts @ g @ wts)


# ---------------------------------------------------------------------------
# Clifford purification (stim)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliffordPurification:
    """Entropies of one Clifford circuit and one mask, in bits, per layer.

    Attributes:
        system: ``S(rho_sys)`` after each of the ``T`` measured layers, for a
            system that starts maximally mixed. ``S(rho_sys) / N`` is the
            entropy of the mid-circuit effect normalised by its trace.
        reference: entropy of a reference qubit entangled with the middle
            site after ``scramble`` unmeasured layers, after each measured
            layer.
    """

    system: np.ndarray
    reference: np.ndarray


def _clifford_layer(
    n: int, gate_ids: Sequence[int], mask_row: np.ndarray | None, offset: int
) -> str:
    programs = clifford_gate_programs()
    lines = [
        programs[g].format(a=a + offset, b=a + 1 + offset)
        for g, a in zip(gate_ids, brick_pairs(n), strict=True)
    ]
    if mask_row is not None:
        sites = np.flatnonzero(mask_row)
        if sites.size:
            lines.append("M " + " ".join(str(int(s) + offset) for s in sites))
    return "\n".join(lines)


def clifford_purification(
    n: int,
    masks: np.ndarray,
    rng: np.random.Generator,
    scramble: int | None = None,
) -> CliffordPurification:
    """Mixed-state and reference-qubit entropies of a Clifford brickwork.

    The same gates and mask serve both probes. Gates are drawn uniformly from
    :func:`~QELM.models.circuits.clifford.clifford_group_2q`.

    Args:
        n: Number of system qubits.
        masks: Boolean ``(T, n)``, the mid-circuit measurements.
        rng: Random generator for the gates.
        scramble: Unmeasured layers before the reference qubit is coupled in
            (default ``2 n``), with their own gates.

    Returns:
        The entropies after each measured layer.
    """
    masks = np.asarray(masks, dtype=bool)
    depth = masks.shape[0]
    ngates = len(brick_pairs(n))
    size = len(clifford_group_2q())
    gates = rng.integers(size, size=(depth, ngates))
    t0 = 2 * n if scramble is None else scramble
    pre = rng.integers(size, size=(t0, ngates))

    # Mixed state: system 0..n-1, each in a Bell pair with ancilla n + i.
    sim = stim.TableauSimulator()
    sim.do_circuit(stim.Circuit("\n".join(f"H {i}\nCX {i} {n + i}" for i in range(n))))
    ancillas = list(range(n, 2 * n))
    system = np.zeros(depth)
    for t in range(depth):
        sim.do_circuit(stim.Circuit(_clifford_layer(n, gates[t], masks[t], 0)))
        system[t] = stabilizer_entropy(stabilizer_matrix(sim), ancillas)

    # Reference qubit: qubit 0 is R, system site i is qubit i + 1. R starts
    # in a Bell pair with the middle site, then the unmeasured layers scramble.
    sim = stim.TableauSimulator()
    sim.do_circuit(stim.Circuit(f"I {n}\nH 0\nCX 0 {n // 2 + 1}"))
    for t in range(t0):
        sim.do_circuit(stim.Circuit(_clifford_layer(n, pre[t], None, 1)))
    reference = np.zeros(depth)
    for t in range(depth):
        sim.do_circuit(stim.Circuit(_clifford_layer(n, gates[t], masks[t], 1)))
        reference[t] = stabilizer_entropy(stabilizer_matrix(sim), [0])
    return CliffordPurification(system=system, reference=reference)


# ---------------------------------------------------------------------------
# Haar reference qubit (state vector)
# ---------------------------------------------------------------------------


def haar_reference_entropy(
    n: int,
    masks: np.ndarray,
    rng: np.random.Generator,
    scramble: int | None = None,
) -> np.ndarray:
    """Entropy of a reference qubit after each measured layer, Haar circuit.

    The reference qubit starts in a Bell pair with the middle site, the
    other sites in ``|0>``. ``scramble`` unmeasured layers (default ``n``)
    follow, then the measured layers, outcomes drawn with the Born rule.
    Fresh gates are drawn from ``rng`` for every call.

    Returns:
        ``S(rho_R)`` in bits after each of the ``T`` measured layers.
    """
    masks = np.asarray(masks, dtype=bool)
    depth = masks.shape[0]
    t0 = n if scramble is None else scramble
    pairs = brick_pairs(n)
    m = n + 1
    # Qubit 0 is R; system site i is qubit i + 1. Bell pair of R and the
    # middle site, (|00> + |11>) / sqrt(2), the other sites in |0>.
    psi = np.zeros(2**m, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[(1 << n) | (1 << (n - 1 - n // 2))] = 1 / np.sqrt(2)
    for _ in range(t0):
        for site in pairs:
            psi = apply_gate(psi, m, site + 1, haar_gate(rng))
    out = np.zeros(depth)
    for t in range(depth):
        for site in pairs:
            psi = apply_gate(psi, m, site + 1, haar_gate(rng))
        for site in np.flatnonzero(masks[t]).tolist():
            p0 = prob_zero(psi, m, int(site) + 1)
            o = 0 if rng.random() < p0 else 1
            psi = project(psi, m, int(site) + 1, o)
            psi /= np.linalg.norm(psi)
        r = psi.reshape(2, -1)
        rho = r @ r.conj().T
        lam = np.clip(np.linalg.eigvalsh(rho), 1e-15, 1.0)
        out[t] = float(-(lam * np.log2(lam)).sum())
    return out
