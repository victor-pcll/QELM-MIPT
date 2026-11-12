"""Clifford QELM: the MIPT in the QELM protocol and its ridge readout (issue #337).

Circuit: the periodic brickwork of random two-qubit Clifford gates of
:mod:`QELM.models.circuits.stabilizer` (Gullans & Huse), one half-layer per
step, each system site measured in ``Z`` with probability ``p`` after the
gates of every step, then the final ``Z`` readout of every site. The QELM
protocol encodes the input at ``t = 0`` on every site; there is no
scrambling prelude.

**Step 1, the transition.** Each system site ``i`` starts in a Bell pair
``(|00> + |11>) / sqrt(2)`` with its own reference qubit ``N + i``. After the
monitored steps, the reduced state of the references is ``E_m^T / Tr E_m``,
with ``E_m`` the effect of the record. For a stabilizer state:

- the entropy of all the references is the number of input bits that the
  record leaves unread (``Delta``, the purification order parameter);
- the entropy of the middle reference is the reference-qubit probe;
- the stabilizer group ``G_m`` of the references (the operators supported on
  them alone) is the group of the effect. A Pauli string ``a`` without ``Y``
  lies in ``+-G_m`` iff ``v_a(m) = sqrt(w_a) c_a / c_0 = +-sqrt(w_a)``, and
  ``v_a = 0`` otherwise. The single-shot R2 of the monomial target ``a``,
  ``E_m[v_a^2]`` in :mod:`QELM.evaluation.ridge_spectrum`, is then
  ``w_a Pr[a in G_m]``. Membership does not depend on the outcomes, only on
  the gates and the mask.

**Step 2, the ridge readout.** The inputs are products of random Pauli
eigenstates ``|0>, |1>, |+>, |->``. They have the first and second moments
of the angle encoding on the circle (mean ``I / 2``, ``E[x^2] = E[z^2] = 1/2``,
``E[x z] = 0``), so every linear quantity of
:mod:`QELM.evaluation.ridge_spectrum` is unchanged, and the circuit stays a
stabilizer circuit. For a fixed circuit and mask, ``stim`` compiles the
circuit of one input and samples its ``K`` shots at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import stim  # type: ignore[import-untyped]

from QELM.models.circuits.stabilizer import (
    clifford_gate_programs,
    gf2_rank,
    periodic_bonds,
    stabilizer_entropy,
    stabilizer_matrix,
)


@dataclass(frozen=True)
class CliffordQELM:
    """Gates and mask of one monitored Clifford brickwork.

    Attributes:
        gates: Index in :func:`clifford_gate_programs` of each gate, shape
            ``(T, N // 2)``, bonds in the order of :func:`periodic_bonds`.
        measured: ``True`` where a site is measured after step ``t``,
            shape ``(T, N)``.
    """

    gates: np.ndarray
    measured: np.ndarray

    @property
    def N(self) -> int:
        return int(self.measured.shape[1])

    @property
    def T(self) -> int:
        return int(self.measured.shape[0])

    @property
    def n_mid(self) -> int:
        """Number of mid-circuit measurements."""
        return int(self.measured.sum())


def draw_circuit(N: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Gate indices of a brickwork of ``T`` steps, shape ``(T, N // 2)``."""
    periodic_bonds(N, 0)
    return rng.integers(len(clifford_gate_programs()), size=(T, N // 2))


def with_mask(gates: np.ndarray, u: np.ndarray, p: float) -> CliffordQELM:
    """The circuit with the mask ``u < p`` (nested from one ``p`` to the next)."""
    return CliffordQELM(gates=np.asarray(gates), measured=np.asarray(u) < p)


def body_program(circuit: CliffordQELM, final: bool = True) -> str:
    """``stim`` program of the monitored steps on qubits ``0 .. N - 1``.

    Gates of step ``t`` on :func:`periodic_bonds` ``(N, t)``, then ``M`` on
    the measured sites; with ``final``, an ``M`` of every site at the end.
    The measurement results come in this order.
    """
    programs = clifford_gate_programs()
    lines: list[str] = []
    for t in range(circuit.T):
        bonds = periodic_bonds(circuit.N, t)
        for g, (a, b) in zip(circuit.gates[t], bonds, strict=True):
            lines.append(programs[int(g)].format(a=a, b=b))
        sites = np.flatnonzero(circuit.measured[t])
        if sites.size:
            lines.append("M " + " ".join(str(int(k)) for k in sites))
    if final:
        lines.append("M " + " ".join(str(k) for k in range(circuit.N)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: purification probes and the group of the effect
# ---------------------------------------------------------------------------


def _reference_group(stab: np.ndarray, n_sys: int) -> np.ndarray:
    """Generators of the stabilizers supported on the references only.

    Args:
        stab: ``[x | z]`` bits of the ``2 n`` generators of the pure state
            of ``n = 2 n_sys`` qubits (system first).
        n_sys: Number of system qubits.

    Returns:
        ``[x | z]`` bits restricted to the references, shape
        ``(r, 2 n_sys)``.
    """
    n = stab.shape[0]
    sys_cols = list(range(n_sys)) + [n + q for q in range(n_sys)]
    ref_cols = list(range(n_sys, n)) + [n + q for q in range(n_sys, n)]
    A = np.array(stab[:, sys_cols + ref_cols], dtype=bool)
    rank = 0
    rows = A.shape[0]
    for col in range(len(sys_cols)):
        piv = np.flatnonzero(A[rank:, col])
        if piv.size == 0:
            continue
        p = rank + int(piv[0])
        A[[rank, p]] = A[[p, rank]]
        others = np.flatnonzero(A[:, col])
        others = others[others != rank]
        A[others] ^= A[rank]
        rank += 1
        if rank == rows:
            break
    return np.asarray(A[rank:, len(sys_cols) :])


def pauli_bits(n: int, ops: dict[int, str]) -> np.ndarray:
    """``[x | z]`` bits of a Pauli string with ``ops[site]`` in ``X, Z``."""
    out = np.zeros(2 * n, dtype=bool)
    for site, op in ops.items():
        if op not in ("X", "Z"):
            raise ValueError(f"Only X and Z are monomials of the encoding, got {op!r}.")
        if op == "X":
            out[site] = True
        else:
            out[n + site] = True
    return out


def in_group(group: np.ndarray, bits: np.ndarray) -> bool:
    """Whether the Pauli ``bits`` (up to sign) is generated by ``group``."""
    if group.shape[0] == 0:
        return not bits.any()
    return gf2_rank(np.vstack([group, bits])) == gf2_rank(group)


@dataclass(frozen=True)
class Purification:
    """Probes of one circuit.

    Attributes:
        delta: Entropy of all the references after the monitored steps
            (no final readout), in bits.
        s_mid: Entropy of the reference of the middle site, same time.
        member_mid: Membership of each target string in ``G_m``, mid-circuit
            record only.
        member_full: Same with the final readout.
    """

    delta: float
    s_mid: float
    member_mid: dict[str, bool]
    member_full: dict[str, bool]


def purification(
    circuit: CliffordQELM, targets: dict[str, dict[int, str]], seed: int = 0
) -> Purification:
    """Run the circuit on the Bell pairs and read the probes.

    Args:
        circuit: Gates and mask.
        targets: Name to Pauli string (``{site: "X" or "Z"}``) of the
            monomials whose membership in ``G_m`` is tested.
        seed: Seed of the outcomes; no probe depends on it.
    """
    N = circuit.N
    sim = stim.TableauSimulator(seed=seed)
    prep = "\n".join(f"H {N + i}\nCX {N + i} {i}" for i in range(N))
    sim.do_circuit(stim.Circuit(prep))
    sim.do_circuit(stim.Circuit(body_program(circuit, final=False)))
    stab = stabilizer_matrix(sim)
    refs = list(range(N, 2 * N))
    delta = stabilizer_entropy(stab, refs)
    s_mid = stabilizer_entropy(stab, [N + N // 2])
    group = _reference_group(stab, N)
    member_mid = {k: in_group(group, pauli_bits(N, v)) for k, v in targets.items()}
    sim.do_circuit(stim.Circuit("M " + " ".join(str(k) for k in range(N))))
    group = _reference_group(stabilizer_matrix(sim), N)
    member_full = {k: in_group(group, pauli_bits(N, v)) for k, v in targets.items()}
    return Purification(
        delta=delta, s_mid=s_mid, member_mid=member_mid, member_full=member_full
    )


def monomial_weight(ops: dict[int, str]) -> float:
    """``w_a = 2^{-|a|}``, the second moment of the monomial on the circle."""
    return float(0.5 ** len(ops))


# ---------------------------------------------------------------------------
# Step 2: QELM with random Pauli-eigenstate inputs
# ---------------------------------------------------------------------------


def draw_inputs(n_inputs: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Random Pauli eigenstates per site: codes 0, 1, 2, 3 for ``|0>, |1>, |+>, |->``."""
    return rng.integers(4, size=(n_inputs, N))


def bloch_components(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bloch components ``(x, z)`` of the input codes, each in ``{-1, 0, 1}``."""
    codes = np.asarray(codes)
    z = np.where(codes == 0, 1, np.where(codes == 1, -1, 0))
    x = np.where(codes == 2, 1, np.where(codes == 3, -1, 0))
    return x.astype(float), z.astype(float)


def monomial_values(codes: np.ndarray, ops: dict[int, str]) -> np.ndarray:
    """Values of ``prod_i sqrt(2) b_i`` (unit variance, mean 0) per input."""
    x, z = bloch_components(codes)
    out = np.ones(codes.shape[0])
    for site, op in ops.items():
        out *= np.sqrt(2.0) * (x[:, site] if op == "X" else z[:, site])
    return out


def degree_one_values(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Unit-variance degree-1 target ``sum_i g_i sqrt(2) x_i + h_i sqrt(2) z_i``.

    Args:
        codes: Input codes, shape ``(S, N)``.
        weights: ``(2, N)``: the ``g`` (X) and ``h`` (Z) weights.
    """
    x, z = bloch_components(codes)
    norm = np.sqrt(np.sum(weights**2))
    return np.asarray(np.sqrt(2.0) * (x @ weights[0] + z @ weights[1]) / norm)


_PREP = {0: "", 1: "X {q}", 2: "H {q}", 3: "X {q}\nH {q}"}


def sample_inputs(
    circuit: CliffordQELM, codes: np.ndarray, shots: int, seed: int
) -> np.ndarray:
    """Draw ``shots`` records of each input, with the Born rule (``stim``).

    Returns:
        ``int8`` array ``(S, shots, n_mid + N)``, ``+1`` for outcome 0 and
        ``-1`` for outcome 1, in the order of :func:`body_program`.
    """
    body = stim.Circuit(body_program(circuit, final=True))
    width = circuit.n_mid + circuit.N
    out = np.empty((codes.shape[0], shots, width), dtype=np.int8)
    for s, row in enumerate(codes):
        prep = "\n".join(
            _PREP[int(c)].format(q=q) for q, c in enumerate(row) if int(c) != 0
        )
        full = stim.Circuit(prep) + body
        bits = full.compile_sampler(seed=seed + s).sample(shots)
        out[s] = 1 - 2 * bits.astype(np.int8)
    return out


def empirical_spectrum(
    shots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signal and noise covariances of the features, from ``K_max`` shots.

    Args:
        shots: ``(S, K_max, d)`` single shots.

    Returns:
        ``(signal, noise, means)``: ``noise = E_x[Cov(M | x)]`` (unbiased),
        ``signal = Cov_x(mean) - noise / K_max`` (the covariance of the exact
        means, unbiased), and the per-input means ``(S, d)``.
    """
    s, kmax, d = shots.shape
    means = shots.mean(axis=1, dtype=np.float64)
    noise = np.zeros((d, d))
    for start in range(0, s, 64):
        x = shots[start : start + 64].astype(np.float32)
        c = (x - means[start : start + 64, None, :].astype(np.float32)).reshape(-1, d)
        noise += (c.T @ c).astype(np.float64)
    noise /= s * (kmax - 1)
    signal = np.cov(means, rowvar=False, ddof=1) - noise / kmax
    return np.asarray(signal), np.asarray(noise), np.asarray(means)


def population_r2(
    signal: np.ndarray, noise: np.ndarray, cov: np.ndarray, shots: float
) -> float:
    """``c^T (Sigma + Noise / K)^+ c`` for a unit-variance target."""
    a = signal if np.isinf(shots) else signal + noise / shots
    return float(cov @ np.linalg.pinv(a, rcond=1e-8, hermitian=True) @ cov)


def targets_for(N: int) -> dict[str, dict[int, str]]:
    """The monomial targets: Z, X of the middle site, ZZ and XX with its neighbour."""
    mid = N // 2
    return {
        "z_mid": {mid: "Z"},
        "x_mid": {mid: "X"},
        "zz_mid": {mid: "Z", mid + 1: "Z"},
        "xx_mid": {mid: "X", mid + 1: "X"},
    }


def mean_and_sem(values: Sequence[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem
