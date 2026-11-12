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
    """Random Pauli eigenstates per site: codes 0-3 for ``|0>, |1>, |+>, |->``."""
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


# ---------------------------------------------------------------------------
# Step 2 at K = infinity: the ridge readout is the group of the effect
# ---------------------------------------------------------------------------
#
# With infinitely many shots, the features of an input are the Born
# probabilities Pr(m | x) = Tr[E_m rho(x)] of the records. For a stabilizer
# circuit, E_m = Tr(E_m) 2^{-r} sum_{g in G} s_g(m) g^T: the Pauli support G
# of the effect is a group of 2^r strings that does not depend on m, only
# the signs s_g(m) = +-1 do, and they are characters of the record. Hence:
#
# - Span{E_m}_m is the span of the strings of G, of dimension 2^r, with
#   r = N - Delta (r = N with the final readout);
# - a string g containing Y gives Tr[g rho(x)] = 0 for every input of the
#   XZ-plane encoding, so the function space of the features is spanned by
#   the monomials B_g of the Y-free strings of G (the visible strings);
# - the population covariance of the features is diagonal in that basis,
#   with eigenvalue proportional to w_g = E[B_g^2] = 2^{-|g|};
# - the population ridge with lambda -> 0+ is the projector on that span:
#   the R2 of a unit-variance target sum_a c_a B_a / sqrt(w_a) is
#   sum_{a in G} c_a^2 / sum_a c_a^2, and E_perp = 1 - R2. For a monomial
#   target, E_perp is 0 or 1.
#
# The elements of G supported on a set A of references form a subgroup of
# rank k_A = r - rank(G restricted to the complement of A). For the pure
# state of system and references, k_A = |A| - S_A, with S_A the
# entanglement entropy of A.


@dataclass(frozen=True)
class EffectGroups:
    """Groups of the effect of one circuit, as ``[x | z]`` generator rows.

    Attributes:
        mid: Generators of ``G`` for the mid-circuit record only,
            shape ``(N - Delta, 2 N)``.
        full: Same with the final readout, shape ``(N, 2 N)``.
    """

    mid: np.ndarray
    full: np.ndarray


def effect_groups(circuit: CliffordQELM, seed: int = 0) -> EffectGroups:
    """Groups of the effect, mid-circuit record and full record.

    Args:
        circuit: Gates and mask.
        seed: Seed of the outcomes; the groups do not depend on it.
    """
    N = circuit.N
    sim = stim.TableauSimulator(seed=seed)
    prep = "\n".join(f"H {N + i}\nCX {N + i} {i}" for i in range(N))
    sim.do_circuit(stim.Circuit(prep))
    sim.do_circuit(stim.Circuit(body_program(circuit, final=False)))
    mid = _reference_group(stabilizer_matrix(sim), N)
    sim.do_circuit(stim.Circuit("M " + " ".join(str(k) for k in range(N))))
    full = _reference_group(stabilizer_matrix(sim), N)
    return EffectGroups(mid=mid, full=full)


class GroupBasis:
    """Reduced row echelon basis of a group of Pauli strings (up to sign).

    Args:
        generators: ``[x | z]`` rows, shape ``(r, 2 n)``; they need not be
            independent.
    """

    def __init__(self, generators: np.ndarray) -> None:
        A = np.array(generators, dtype=bool)
        if A.ndim != 2:
            raise ValueError("generators must be a 2-D array of [x | z] bits.")
        pivots: list[int] = []
        rank = 0
        for col in range(A.shape[1]):
            if rank == A.shape[0]:
                break
            piv = np.flatnonzero(A[rank:, col])
            if piv.size == 0:
                continue
            p = rank + int(piv[0])
            A[[rank, p]] = A[[p, rank]]
            others = np.flatnonzero(A[:, col])
            others = others[others != rank]
            A[others] ^= A[rank]
            pivots.append(col)
            rank += 1
        self.rows = A[:rank]
        self.pivots = np.array(pivots, dtype=int)
        self.n = A.shape[1] // 2

    @property
    def rank(self) -> int:
        return int(self.rows.shape[0])

    def contains(self, strings: np.ndarray) -> np.ndarray:
        """Membership of each ``[x | z]`` row of ``strings`` (up to sign)."""
        v = np.array(np.atleast_2d(strings), dtype=bool)
        for row, col in zip(self.rows, self.pivots, strict=True):
            hit = v[:, col]
            v[hit] ^= row
        return np.asarray(~v.any(axis=1))

    def local_rank(self, region: Sequence[int]) -> int:
        """Rank ``k_A`` of the subgroup supported on the sites of ``region``."""
        inside = np.zeros(self.n, dtype=bool)
        inside[list(region)] = True
        out = np.flatnonzero(~inside)
        cols = np.concatenate([out, self.n + out])
        if cols.size == 0 or self.rank == 0:
            return self.rank
        return self.rank - gf2_rank(self.rows[:, cols])

    def subgroup(self, region: Sequence[int]) -> np.ndarray:
        """Generators of the subgroup supported on ``region``."""
        inside = np.zeros(self.n, dtype=bool)
        inside[list(region)] = True
        out = np.flatnonzero(~inside)
        cols = np.concatenate([out, self.n + out])
        A = np.concatenate([self.rows[:, cols], self.rows], axis=1)
        rank = 0
        for col in range(cols.size):
            if rank == A.shape[0]:
                break
            piv = np.flatnonzero(A[rank:, col])
            if piv.size == 0:
                continue
            p = rank + int(piv[0])
            A[[rank, p]] = A[[p, rank]]
            others = np.flatnonzero(A[:, col])
            others = others[others != rank]
            A[others] ^= A[rank]
            rank += 1
        return np.asarray(A[rank:, cols.size :])


def enumerate_group(generators: np.ndarray) -> np.ndarray:
    """All ``2^r`` elements of the group, ``[x | z]`` rows, identity first."""
    gens = np.asarray(generators, dtype=bool)
    if gens.shape[0] > 22:
        raise ValueError(f"2^{gens.shape[0]} elements is too many to enumerate.")
    out = np.zeros((1, gens.shape[1]), dtype=bool)
    for g in gens:
        out = np.concatenate([out, out ^ g], axis=0)
    return out


def visible_weights(elements: np.ndarray) -> np.ndarray:
    """Weights ``|g|`` of the Y-free, non-identity strings among ``elements``."""
    e = np.asarray(elements, dtype=bool)
    n = e.shape[1] // 2
    x, z = e[:, :n], e[:, n:]
    weight = (x | z).sum(axis=1)
    keep = ~(x & z).any(axis=1) & (weight > 0)
    return np.asarray(weight[keep])


def interval(start: int, length: int, N: int) -> list[int]:
    """Sites ``start, ..., start + length - 1`` of the periodic chain."""
    return [(start + k) % N for k in range(length)]


def local_visible_weights(basis: GroupBasis, span: int) -> np.ndarray:
    """Weights of the distinct visible strings of ``G`` within ``span`` sites.

    Every visible string whose support fits in an interval of ``span``
    sites is counted once.
    """
    N = basis.n
    seen: set[bytes] = set()
    weights: list[int] = []
    for start in range(N):
        sub = basis.subgroup(interval(start, span, N))
        if sub.shape[0] == 0:
            continue
        elems = enumerate_group(sub)
        x, z = elems[:, :N], elems[:, N:]
        keep = ~(x & z).any(axis=1) & (x | z).any(axis=1)
        for row, w in zip(elems[keep], (x | z)[keep].sum(axis=1), strict=True):
            key = np.packbits(row).tobytes()
            if key not in seen:
                seen.add(key)
                weights.append(int(w))
    return np.array(weights, dtype=int)


def range_targets(N: int, ranges: Sequence[int]) -> dict[str, np.ndarray]:
    """Monomial targets on every site, as ``[x | z]`` rows per family.

    Families: ``Z``, ``X`` (one site), ``ZZ_l``, ``XX_l`` and ``ZX_l`` (two
    sites at distance ``l``, for each ``l`` in ``ranges``), each with ``N``
    rows (one per first site).
    """
    out: dict[str, np.ndarray] = {}
    for name, ops in (("Z", "Z"), ("X", "X")):
        out[name] = np.stack([pauli_bits(N, {i: ops}) for i in range(N)])
    for ell in ranges:
        for name in ("ZZ", "XX", "ZX"):
            rows = [
                pauli_bits(N, {i: name[0], (i + ell) % N: name[1]}) for i in range(N)
            ]
            out[f"{name}_{ell}"] = np.stack(rows)
    return out


def effective_dimension(weights: np.ndarray, lambdas: Sequence[float]) -> np.ndarray:
    """``d_eff(lambda) = sum_g w_g / (w_g + lambda)``, with ``w_g = 2^{-|g|}``."""
    w = 0.5 ** np.asarray(weights, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    return np.asarray((w[None, :] / (w[None, :] + lam[:, None])).sum(axis=1))


def interval_entropies(
    generators: np.ndarray, max_len: int | None = None
) -> np.ndarray:
    """Entropy ``S(A)`` of every interval of a pure stabilizer state.

    For ``N`` independent generators of a pure state of ``N`` qubits,
    ``S_A = rank(G restricted to A) - |A|``. The rank is updated site by
    site, the two columns of each new site being reduced against an echelon
    basis kept as Python integers (one bit per generator).

    Args:
        generators: ``[x | z]`` rows, shape ``(N, 2 N)``, independent (the
            full-record group of :func:`effect_groups`).
        max_len: Longest interval (default ``N``).

    Returns:
        Array ``(N, max_len + 1)``: ``S`` of the interval of length ``l``
        starting at each site, ``l = 0 .. max_len``.
    """
    g = np.asarray(generators, dtype=bool)
    n = g.shape[1] // 2
    if g.shape[0] != n:
        raise ValueError(f"A pure state needs {n} generators, got {g.shape[0]}.")
    max_len = n if max_len is None else max_len
    weights = [1 << k for k in range(n)]

    def column(c: int) -> int:
        return sum(w for w, bit in zip(weights, g[:, c], strict=True) if bit)

    cols = [(column(q), column(n + q)) for q in range(n)]
    out = np.zeros((n, max_len + 1))
    for start in range(n):
        pivots: dict[int, int] = {}
        for ell in range(1, max_len + 1):
            for v in cols[(start + ell - 1) % n]:
                while v:
                    top = v.bit_length() - 1
                    if top not in pivots:
                        pivots[top] = v
                        break
                    v ^= pivots[top]
            out[start, ell] = len(pivots) - ell
    return out


# ---------------------------------------------------------------------------
# The ridge on local features: an exact Fourier filter (issue #341)
# ---------------------------------------------------------------------------
#
# The local feature of a measured bit j is its marginal mean
# E[(-1)^{m_j} | x] = Tr[rho(x) O_j]. O_j is the Z of the bit, back-propagated
# through the gates (U^dagger . U) and through the dephasing channels of the
# earlier measurements: a Pauli string goes through the dephasing of site k
# if it commutes with Z_k and is killed otherwise. So O_j is one Pauli string
# (up to sign) or zero, the Fourier transfer matrix C of the local readout
# is a selection matrix, and the population covariance of the features,
# Sigma = C W C^T, is diagonal in the distinct visible strings, with
# eigenvalue m_a w_a for a string present m_a times.

_HEISENBERG: np.ndarray | None = None


def heisenberg_table() -> np.ndarray:
    """``U^dagger P U`` of every two-qubit Clifford on the 16 Paulis.

    Pauli codes per qubit: 0 ``I``, 1 ``X``, 2 ``Z``, 3 ``Y``; a two-qubit
    Pauli is ``4 c_a + c_b`` with ``a`` the most significant qubit of the
    gate (the convention of :func:`clifford_group_2q`).

    Returns:
        Integer array ``(11520, 16)``, cached.
    """
    global _HEISENBERG
    if _HEISENBERG is None:
        from QELM.models.circuits.clifford import clifford_group_2q

        one = [
            np.eye(2),
            np.array([[0, 1], [1, 0]]),
            np.diag([1.0, -1.0]),
            np.array([[0, -1j], [1j, 0]]),
        ]
        paulis = np.array([np.kron(a, b) for a in one for b in one], dtype=complex)
        gates = np.array(clifford_group_2q())
        conj = np.einsum("gji,pjk,gkl->gpil", gates.conj(), paulis, gates)
        overlap = np.abs(np.einsum("qji,gpji->gpq", paulis.conj(), conj)) / 4
        _HEISENBERG = np.asarray(np.argmax(overlap, axis=2))
    return _HEISENBERG


def local_strings(circuit: CliffordQELM) -> tuple[np.ndarray, np.ndarray]:
    """Back-propagated Pauli string of every measured bit.

    Args:
        circuit: Gates and mask; the bits are in the order of
            :func:`body_program` (mid-circuit, then the final readout).

    Returns:
        ``(strings, alive)``: ``[x | z]`` bits of ``O_j``, shape
        ``(n_mid + N, 2 N)``, and whether ``O_j`` survives the dephasing
        (``Tr[rho O_j]`` is then the feature, up to sign; zero otherwise).
    """
    table = heisenberg_table()
    n, t_steps = circuit.N, circuit.T
    n_bits = circuit.n_mid + n
    code = np.zeros((n_bits, n), dtype=np.int64)  # 0 I, 1 X, 2 Z, 3 Y
    alive = np.ones(n_bits, dtype=bool)
    active = np.zeros(n_bits, dtype=bool)
    start = np.cumsum([0, *circuit.measured.sum(axis=1)])
    final = np.arange(circuit.n_mid, n_bits)
    code[final, np.arange(n)] = 2
    active[final] = True
    for t in reversed(range(t_steps)):
        sites = np.flatnonzero(circuit.measured[t])
        # Dephasing of the measurements of step t on the later bits.
        rows = np.flatnonzero(active)
        for k in sites:
            alive[rows] &= (code[rows, k] == 0) | (code[rows, k] == 2)
        # The bits measured at step t start here.
        new = np.arange(start[t], start[t + 1])
        code[new, sites] = 2
        active[new] = True
        rows = np.flatnonzero(active & alive)
        for g, (a, b) in zip(circuit.gates[t], periodic_bonds(n, t), strict=True):
            out = table[int(g), 4 * code[rows, a] + code[rows, b]]
            code[rows, a], code[rows, b] = out // 4, out % 4
    strings = np.concatenate([(code == 1) | (code == 3), code >= 2], axis=1)
    return strings, alive


def bit_times(circuit: CliffordQELM) -> tuple[np.ndarray, np.ndarray]:
    """Step (``T`` for the final readout) and site of every measured bit."""
    steps, sites = np.nonzero(circuit.measured)
    n = circuit.N
    return (
        np.concatenate([steps, np.full(n, circuit.T)]),
        np.concatenate([sites, np.arange(n)]),
    )


def parity_strings(
    circuit: CliffordQELM, sets: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Back-propagated string of the parity ``prod_{j in S} (-1)^{m_j}``.

    The mean ``E[prod (-1)^{m_j} | x]`` is ``Tr[rho(x) O_S]``, with ``O_S``
    one Pauli string (up to sign) or zero. Going backwards, the measurement
    of site ``k`` at step ``t`` kills every later operator that does not
    commute with ``Z_k``, then multiplies by ``Z_k`` the parities that
    contain that bit. :func:`local_strings` is the case of single bits.

    Args:
        circuit: Gates and mask.
        sets: Bit indices of each parity (order of :func:`body_program`).

    Returns:
        ``(strings, alive)`` as in :func:`local_strings`, one row per set.
    """
    table = heisenberg_table()
    n = circuit.N
    steps, sites = bit_times(circuit)
    n_sets = len(sets)
    x = np.zeros((n_sets, n), dtype=bool)
    z = np.zeros((n_sets, n), dtype=bool)
    alive = np.ones(n_sets, dtype=bool)
    started = np.zeros(n_sets, dtype=bool)
    # For each step, (set index, site) pairs of the bits measured there.
    events: dict[int, list[tuple[int, int]]] = {}
    for s, bits in enumerate(sets):
        for j in bits:
            events.setdefault(int(steps[j]), []).append((s, int(sites[j])))

    def measure(t: int) -> None:
        here = events.get(t, [])
        rows = np.flatnonzero(started & alive)
        if t < circuit.T:
            for site in np.flatnonzero(circuit.measured[t]):
                alive[rows] &= ~x[rows, site]
        for s, k in here:
            z[s, k] ^= True
            started[s] = True

    measure(circuit.T)
    for t in reversed(range(circuit.T)):
        measure(t)
        rows = np.flatnonzero(started & alive)
        code = x[rows].astype(np.int64) + 2 * z[rows].astype(np.int64)
        for g, (a, b) in zip(circuit.gates[t], periodic_bonds(n, t), strict=True):
            out = table[int(g), 4 * code[:, a] + code[:, b]]
            code[:, a], code[:, b] = out // 4, out % 4
        x[rows], z[rows] = (code % 2).astype(bool), (code >= 2)
    strings = np.concatenate([x, z], axis=1)
    return strings, alive & started
