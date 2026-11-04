"""Exact bounds on what the late rounds of a monitored circuit can tell (#288).

The setting of ``experiments/nonlinear_M`` and ``experiments/early_layers``: a
reference qubit ``R`` in the state ``(|0>_R |a> + |1>_R |b>) / sqrt(2)`` with
the chain, a fixed brickwork circuit on the chain, and after the gates of
each round a Z measurement of every site with probability ``p``. The label
``Y`` is the Z outcome of ``R``. Since ``R`` never evolves, measuring it at the
end is the same as measuring it first: the chain starts in ``|a>`` (``Y = +1``)
or ``|b>`` (``Y = -1``) with probability 1/2 each.

The tools of this module, each the computable side of one statement of
``documentation/notes/early-layers-proof.md``:

- :func:`averaged_states`: the chain states ``rho_t^+`` and ``rho_t^-``
  averaged over every earlier measurement (where and what), right after the
  gates of round ``t``. They evolve under a fixed quantum channel, so
  ``D_t = ||rho_t^+ - rho_t^-||_1 / 2`` never grows with ``t`` (data
  processing); :func:`suffix_bounds` returns ``D_t`` and the exact linear
  signal ``c_{t,i} = E[m_{t,i} Y] = (p / 2) Tr(Z_i (rho_t^+ - rho_t^-))``.
  No decoder of the rounds ``>= t`` alone beats ``(1 + D_t) / 2``, and
  ``|c_{t,i}| <= p D_t``.
- :func:`reference_helstrom`: along one trajectory, ``<Z_R>`` given the
  record so far and the Helstrom value ``sqrt(1 - 4 |<A|B>|^2)`` of the joint
  state, ``A = <0|_R psi``, ``B = <1|_R psi``. Averaged over trajectories,
  ``(1 + E[H_k]) / 2`` bounds every decoder of the whole record, rounds after
  ``k`` included, while ``(1 + E|<Z_R>_k|) / 2`` is the Bayes accuracy of the
  first ``k`` rounds.
- :func:`haar_pauli_chain`: the average over Haar U(4) gates (a unitary
  2-design) of the squared Pauli coefficients of ``rho_t^+ - rho_t^-``,
  grouped by type (``I``, ``Z``, ``X`` or ``Y`` on each site): an exact
  Markov chain on ``3**N`` configurations. It gives ``E[c_{t,i}^2]`` and the
  bound ``E[D_t] <= sqrt(mu_t) / 2`` from the total mass ``mu_t``;
  :func:`haar_mass_rate` is the per-round contraction ``g(p)`` of
  ``mu_t``, ``mu_{t+1} <= g(p) mu_t``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from QELM.models.circuits.base import apply_two_qubit_gate

# A layer of the circuit: pairs of qubits and their 4 x 4 gates.
Layer = Sequence[tuple[int, int, np.ndarray]]

# Pauli types on one site in the Haar chain.
TYPE_I, TYPE_Z, TYPE_XY = 0, 1, 2
# Number of two-site Paulis of each type pair, among the 15 non-identity ones:
# (I, Z), (Z, I), (Z, Z) once each; a type with X or Y counts its two Paulis.
_PAIR_COUNTS = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
        [2.0, 2.0, 4.0],
    ]
)
_PAIR_WEIGHTS = _PAIR_COUNTS / 15.0


def split_reference_state(psi0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chain states ``|a>`` and ``|b>`` of ``(|0>_R |a> + |1>_R |b>) / sqrt 2``.

    ``R`` is qubit 0, the most significant bit of ``psi0``.

    Returns:
        ``(a, b)``, each normalized.

    Raises:
        ValueError: If ``psi0`` does not have that form (one branch empty,
            or weights other than 1/2 each).
    """
    psi0 = np.asarray(psi0, dtype=complex)
    half = psi0.size // 2
    if psi0.ndim != 1 or 2 * half != psi0.size or half < 1:
        raise ValueError(f"psi0 must be a state of R and the chain, got {psi0.shape}.")
    a, b = psi0[:half], psi0[half:]
    na, nb = float(np.vdot(a, a).real), float(np.vdot(b, b).real)
    if not (np.isclose(na, 0.5, atol=1e-9) and np.isclose(nb, 0.5, atol=1e-9)):
        raise ValueError(f"Both branches of psi0 must weigh 1/2, got {na}, {nb}.")
    return a / np.sqrt(na), b / np.sqrt(nb)


def dephase(rho: np.ndarray, site: int, p: float) -> np.ndarray:
    """Average of a Z measurement of ``site`` made with probability ``p``.

    ``(1 - p) rho + p (P_0 rho P_0 + P_1 rho P_1)``: the elements of ``rho``
    off-diagonal in ``site`` are multiplied by ``1 - p``. ``site`` 0 is the
    most significant bit.
    """
    dim = rho.shape[0]
    n = dim.bit_length() - 1
    rows = (np.arange(dim) >> (n - 1 - site)) & 1
    keep = np.where(rows[:, None] == rows[None, :], 1.0, 1.0 - p)
    return np.asarray(rho * keep)


def averaged_states(
    layers: Sequence[Layer],
    a: np.ndarray,
    b: np.ndarray,
    p: float,
    offset: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory-averaged chain states after the gates of each round.

    Args:
        layers: The rounds of the circuit; each gate acts on qubits
            ``(i, j)`` of the state with ``R``, so on chain sites
            ``(i - offset, j - offset)``.
        a, b: Initial chain states of ``Y = +1`` and ``Y = -1``.
        p: Measurement probability of each site in each round.
        offset: Index of the first chain site in the qubits of the gates
            (the number of ancillas).

    Returns:
        ``(rho_plus, rho_minus)``, each of shape ``(T, 2**N, 2**N)``: entry
        ``t`` is the state right after the gates of round ``t + 1``, averaged
        over the measurements of rounds ``1 .. t``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    dim = a.size
    n = dim.bit_length() - 1
    out = []
    for vec in (a, b):
        rho = np.outer(vec, vec.conj())
        states = np.zeros((len(layers), dim, dim), dtype=complex)
        for t, layer in enumerate(layers):
            if t > 0:
                for site in range(n):
                    rho = dephase(rho, site, p)
            for i, j, gate in layer:
                rho = apply_two_qubit_gate(rho, gate, i - offset, j - offset)
            states[t] = rho
        out.append(states)
    return out[0], out[1]


def z_expectations(rho: np.ndarray) -> np.ndarray:
    """``Tr(Z_i rho)`` for every site ``i`` (site 0 the most significant bit).

    ``rho`` may carry leading batch axes: shape ``(..., 2**N, 2**N)``.
    """
    dim = rho.shape[-1]
    n = dim.bit_length() - 1
    diag = np.real(np.diagonal(rho, axis1=-2, axis2=-1))
    bits = (np.arange(dim)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    signs = 1.0 - 2.0 * bits
    return np.asarray(diag @ signs)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``||rho - sigma||_1 / 2`` (batched over leading axes)."""
    eig = np.linalg.eigvalsh(rho - sigma)
    return np.asarray(0.5 * np.sum(np.abs(eig), axis=-1))


def suffix_bounds(
    layers: Sequence[Layer],
    a: np.ndarray,
    b: np.ndarray,
    p: float,
    offset: int = 1,
) -> dict[str, np.ndarray]:
    """Trace distance ``D_t`` and exact linear signal ``c_{t,i}`` per round.

    Returns:
        ``D``, shape ``(T,)``: ``||rho_t^+ - rho_t^-||_1 / 2``, the largest
        advantage over chance of any decoder of rounds ``t .. T``
        (accuracy ``<= (1 + D_t) / 2``); ``c``, shape ``(T, N)``:
        ``E[m_{t,i} Y]``, the population covariance of each entry of the
        record with the label; ``charge``, shape ``(T,)``:
        ``Tr(Q (rho_t^+ - rho_t^-))`` with ``Q = sum_i (1 - Z_i) / 2``,
        conserved by U(1) gates.
    """
    rho_p, rho_m = averaged_states(layers, a, b, p, offset)
    diff_z = z_expectations(rho_p) - z_expectations(rho_m)
    return {
        "D": trace_distance(rho_p, rho_m),
        "c": 0.5 * p * diff_z,
        "charge": -0.5 * diff_z.sum(axis=1),
    }


def reference_helstrom(psi: np.ndarray) -> tuple[float, float]:
    """``<Z_R>`` and the Helstrom value of a joint state of ``R`` and the chain.

    With ``A = <0|_R psi`` and ``B = <1|_R psi`` (unnormalized),
    ``<Z_R> = |A|^2 - |B|^2`` and the operator
    ``Tr_R[(Z_R x 1) |psi><psi|] = |A><A| - |B><B|`` has trace norm
    ``sqrt(1 - 4 |<A|B>|^2)``: the largest ``2 P(correct) - 1`` of any
    measurement of the chain that guesses ``Y`` (Helstrom). It is never
    below ``|<Z_R>|``.

    Returns:
        ``(r_z, H)``.
    """
    half = psi.size // 2
    A, B = psi[:half], psi[half:]
    r_z = float(np.vdot(A, A).real - np.vdot(B, B).real)
    overlap = float(abs(np.vdot(A, B)) ** 2)
    H = float(np.sqrt(max(0.0, 1.0 - 4.0 * overlap)))
    return r_z, max(H, abs(r_z))


def pauli_type_mass(delta: np.ndarray) -> np.ndarray:
    """Squared Pauli coefficients of an operator, summed by type per site.

    ``delta = sum_P a_P P / 2**N`` with ``a_P = Tr(P delta)``. The result,
    shape ``(3,) * N``, holds ``sum a_P^2`` over the strings ``P`` of each
    type configuration (:data:`TYPE_I`, :data:`TYPE_Z`, :data:`TYPE_XY`).
    """
    dim = delta.shape[0]
    n = dim.bit_length() - 1
    # Pauli basis: coefficient a_P = sum_{rs} P_{sr} delta_{rs}.
    paulis = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=complex,
    )
    tensor = delta.reshape((2,) * (2 * n))
    # Pair the row and column axes of each site: (r0, c0, r1, c1, ...).
    order = [k for site in range(n) for k in (site, n + site)]
    tensor = tensor.transpose(order)
    for _site in range(n):
        # Contract the leading (r, c) pair with P_{c r}, put the Pauli index last.
        tensor = np.einsum("rc...,qcr->...q", tensor, paulis)
    squared = np.abs(tensor) ** 2  # axes: Pauli index of sites 0 .. N - 1
    # Pauli index order I, X, Y, Z -> types I, XY, XY, Z.
    collapse = np.zeros((4, 3))
    collapse[0, TYPE_I] = collapse[1, TYPE_XY] = collapse[2, TYPE_XY] = 1.0
    collapse[3, TYPE_Z] = 1.0
    for site in range(n):
        squared = np.moveaxis(
            np.tensordot(squared, collapse, axes=([site], [0])), -1, site
        )
    return np.asarray(squared)


def _haar_gate(mass: np.ndarray, i: int, j: int) -> np.ndarray:
    """Haar average of one U(4) gate on sites ``(i, j)`` of a type mass."""
    moved = np.moveaxis(mass, (i, j), (0, 1))
    identity = moved[0, 0].copy()
    total = moved.sum(axis=(0, 1)) - identity
    out = _PAIR_WEIGHTS[:, :, None] * total.reshape(1, 1, -1)
    out = out.reshape((3, 3) + total.shape)
    out[0, 0] = identity
    return np.moveaxis(out, (0, 1), (i, j))


def _dephase_mass(mass: np.ndarray, p: float) -> np.ndarray:
    """Every site measured with probability ``p``: ``X``, ``Y`` get ``(1 - p)^2``."""
    factor = np.array([1.0, 1.0, (1.0 - p) ** 2])
    n = mass.ndim
    for site in range(n):
        shape = [1] * n
        shape[site] = 3
        mass = mass * factor.reshape(shape)
    return mass


def haar_pauli_chain(
    initial_mass: np.ndarray,
    rounds: Sequence[Sequence[tuple[int, int]]],
    p: float,
) -> dict[str, np.ndarray]:
    """Haar average of the type mass of ``rho_t^+ - rho_t^-`` round by round.

    Each gate is drawn independently from the Haar measure on U(4), a
    2-design: the average of ``(U x U) (P x Q) (U x U)^dagger`` is zero for
    ``P != Q`` and spreads a non-identity ``P`` uniformly over the 15
    non-identity Paulis of the pair. The squared Pauli coefficients thus
    evolve exactly by a linear Markov chain on the type configurations.

    Args:
        initial_mass: :func:`pauli_type_mass` of ``|a><a| - |b><b|``.
        rounds: Per round, the chain-site pairs of its gates, in order.
        p: Measurement probability.

    Returns:
        ``mu``, shape ``(T,)``: total mass ``E[sum_P a_P^2]`` after the gates
        of each round, so ``E[D_t] <= sqrt(mu_t) / 2``; ``single_z``, shape
        ``(T, N)``: ``E[a_{Z_i}^2]``, so ``E[c_{t,i}^2] = (p / 2)^2`` times
        it; ``weight_one``, shape ``(T,)``: the mass on the strings of weight
        one, the only ones a single entry of the record sees.
    """
    mass = np.asarray(initial_mass, dtype=float)
    n = mass.ndim
    T = len(rounds)
    mu = np.zeros(T)
    single_z = np.zeros((T, n))
    weight_one = np.zeros(T)
    for t, pairs in enumerate(rounds):
        if t > 0:
            mass = _dephase_mass(mass, p)
        for i, j in pairs:
            mass = _haar_gate(mass, i, j)
        mu[t] = mass.sum() - mass[(0,) * n]
        for i in range(n):
            z_idx = [TYPE_I] * n
            z_idx[i] = TYPE_Z
            xy_idx = [TYPE_I] * n
            xy_idx[i] = TYPE_XY
            single_z[t, i] = mass[tuple(z_idx)]
            weight_one[t] += mass[tuple(z_idx)] + mass[tuple(xy_idx)]
    return {"mu": mu, "single_z": single_z, "weight_one": weight_one}


def haar_mass_rate(p: float) -> float:
    """Per-round contraction of the Haar-averaged mass, ``g(p)``.

    ``g(p) = (1 + 2 (1 - p)^2) / 3``: every non-identity string meets, in the
    last half-layer of a round, either a gate (whose output is uniform on 15
    Paulis, kept on average with weight ``(3 + 8 q + 4 q^2) / 15``,
    ``q = (1 - p)^2``) or only the free edge sites, uniform over ``Z, X, Y``
    after the first half-layer (weight ``(1 + 2 q) / 3``, the larger).
    For an even ``N`` and an open chain, ``mu_{t+1} <= g(p) mu_t``.
    """
    q = (1.0 - p) ** 2
    return (1.0 + 2.0 * q) / 3.0
