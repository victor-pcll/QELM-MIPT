"""Population R^2 budget of random dense targets, Clifford and Haar (issue #360).

In the population limit (``K = inf`` shots, infinitely many training inputs,
``lambda -> 0+``) the ridge on a set of features is the ``W``-orthogonal
projection of the target on their span, ``W = diag(w_a)`` the weights of the
Bloch monomials. For a target built in the monomial basis,
``y = sum_{a != I} beta_a B_a`` (so ``E_encoding = 0``), and two nested
readouts of the same record,

- ``S``: every record probability (equivalently every parity of the bits),
  feature rows ``C``, projector ``Pi_C``;
- ``F^(k)``: the parities of at most ``k`` bits, rows ``C^(k)``, projector
  ``Pi_C^(k)``, with ``span C^(k) ⊂ span C``,

Pythagoras gives, without cross term,

    1 = E_circuit + E_readout(k) + R^2_pop,

with ``E_circuit = ||(1 - Pi_C) beta||_W^2 / ||beta||_W^2``,
``E_readout(k) = ||(Pi_C - Pi_C^(k)) beta||_W^2 / ||beta||_W^2`` and
``R^2_pop = ||Pi_C^(k) beta||_W^2 / ||beta||_W^2``.

Two routes compute it:

- :func:`dense_budget` (any gate set, ``N <= 4``): exact dense projections in
  whitened Pauli space ``g = W^{1/2} beta``, from the feature matrices of
  :mod:`QELM.evaluation.fourier_framework`. The cross term is computed, not
  assumed.
- :func:`clifford_budget` (Clifford, ``N`` up to about 12): the span of the
  features is spanned by Pauli strings (the visible strings of the effect
  group ``G``, and the strings ``D_k`` of the parities of at most ``k``
  bits), which are ``W``-orthogonal, so each projection keeps the
  coefficients of its strings: ``E_circuit = sum_{a notin G} w_a beta_a^2``,
  ``E_readout(k) = sum_{a in G minus D_k} w_a beta_a^2`` (over ``Var y``).

Target priors (both on the ``3^N - 1`` visible strings of the XZ ensemble):

- ``"iid"``: ``beta_a ~ N(0, 1)`` i.i.d. (homogeneous on the coefficients);
  the share of a string in ``Var y`` is ``w_a = 2^{-|a|}`` on average;
- ``"flat"``: ``sqrt(w_a) beta_a ~ N(0, 1)`` (equal power per string).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation.clifford_ridge import (
    CliffordQELM,
    GroupBasis,
    effect_groups,
    enumerate_group,
    parity_strings,
)

PRIORS = ("iid", "flat")


@dataclass(frozen=True)
class Budget:
    """Budget terms of a batch of targets, each divided by ``Var y``.

    Attributes:
        circuit: ``(n_targets,)`` ``E_circuit``.
        readout: ``(n_k, n_targets)`` ``E_readout(k)``, one row per ``k``.
        r2: ``(n_k, n_targets)`` ``R^2_pop(k)``.
        cross: ``(n_k, n_targets)`` cross term ``2 <(1 - Pi_C) g, (Pi_C -
            Pi_C^(k)) g>`` (zero in exact arithmetic).
    """

    circuit: np.ndarray
    readout: np.ndarray
    r2: np.ndarray
    cross: np.ndarray


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def draw_whitened_targets(
    n: int, n_targets: int, prior: str, rng: np.random.Generator
) -> np.ndarray:
    """Whitened coefficients ``g_a = sqrt(w_a) beta_a`` of random dense targets.

    Args:
        n: Number of qubits.
        n_targets: Number of targets.
        prior: ``"iid"`` or ``"flat"`` (module docstring).
        rng: Random generator.

    Returns:
        ``(n_targets, 4^n - 1)`` array over the strings ``a != I`` in the order
        of :func:`QELM.evaluation.fourier_framework.pauli_codes`; zero on the
        strings with a ``Y``.
    """
    if prior not in PRIORS:
        raise ValueError(f"unknown prior {prior!r}")
    w = ff.input_weights(n, "xz")[1:]
    z = rng.standard_normal((n_targets, w.size))
    scale = np.sqrt(w) if prior == "iid" else (w > 0).astype(float)
    return np.asarray(z * scale)


# ---------------------------------------------------------------------------
# Dense route
# ---------------------------------------------------------------------------


def span_basis(c: np.ndarray, w: np.ndarray, rel_tol: float = 1e-10) -> np.ndarray:
    """Orthonormal rows spanning ``C W^{1/2}`` (columns ``a != I``).

    Same cut-off as :func:`QELM.evaluation.fourier_framework.population_error`.
    """
    a = c[:, 1:] * np.sqrt(w[1:])
    if not np.any(a):
        return np.zeros((0, a.shape[1]))
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    keep = s > max(rel_tol * s[0], 1e-12)
    return np.asarray(vt[keep])


def project_budget(v_s: np.ndarray, v_ks: list[np.ndarray], g: np.ndarray) -> Budget:
    """Budget of whitened targets ``g`` from orthonormal span bases.

    Args:
        v_s: Rows spanning the full feature space (``Pi_C``).
        v_ks: Rows spanning each ``C^(k)`` (nested in ``v_s``).
        g: ``(n_targets, d)`` whitened targets.
    """
    var = np.sum(g**2, axis=1)
    ps = (g @ v_s.T) @ v_s
    out_c = g - ps
    readout, r2, cross = [], [], []
    for v_k in v_ks:
        pk = (g @ v_k.T) @ v_k
        mid = ps - pk
        readout.append(np.sum(mid**2, axis=1) / var)
        r2.append(np.sum(pk**2, axis=1) / var)
        cross.append(2 * np.sum(out_c * mid, axis=1) / var)
    return Budget(
        circuit=np.sum(out_c**2, axis=1) / var,
        readout=np.array(readout),
        r2=np.array(r2),
        cross=np.array(cross),
    )


def dense_budget(
    circuit: ff.Circuit, ks: tuple[int, ...], g: np.ndarray, max_bits: int = 16
) -> Budget:
    """Exact budget on any circuit of :mod:`fourier_framework` (final readout on)."""
    n = circuit.n
    w = ff.input_weights(n, "xz")
    records = ff.enumerate_records(circuit, final=True, max_bits=max_bits)
    c = ff.feature_matrix(records.effects)
    v_s = span_basis(c, w)
    v_ks = [span_basis(ff.parity_features(c, records.bits, k), w) for k in ks]
    return project_budget(v_s, v_ks, g)


# ---------------------------------------------------------------------------
# Clifford route: counting on Pauli strings
# ---------------------------------------------------------------------------


def visible_index(strings: np.ndarray) -> np.ndarray:
    """Index ``sum_i c_i 3^i`` (``I, X, Z = 0, 1, 2``) of Y-free ``[x | z]`` rows."""
    s = np.asarray(strings, dtype=bool)
    n = s.shape[1] // 2
    code = s[:, :n].astype(np.int64) + 2 * s[:, n:].astype(np.int64)
    return np.asarray(code @ (3 ** np.arange(n, dtype=np.int64)))


def visible_strings(n: int) -> np.ndarray:
    """All ``3^n`` Y-free ``[x | z]`` rows, in :func:`visible_index` order."""
    idx = np.arange(3**n)
    code = (idx[:, None] // 3 ** np.arange(n)[None, :]) % 3
    return np.concatenate([code == 1, code == 2], axis=1)


def _keep_visible(s: np.ndarray) -> np.ndarray:
    n = s.shape[1] // 2
    return s[~(s[:, :n] & s[:, n:]).any(axis=1)]


def clifford_masks(
    circuit: CliffordQELM, ks: tuple[int, ...], chunk: int = 200_000
) -> tuple[np.ndarray, np.ndarray]:
    """Membership of every visible string in ``G`` and in each ``D_k``.

    Returns:
        ``(in_g, in_d)``: boolean arrays of shape ``(3^N,)`` and
        ``(len(ks), 3^N)``, in :func:`visible_index` order (index 0 is ``I``).
    """
    n = circuit.N
    gens = GroupBasis(effect_groups(circuit).full).rows
    in_g = np.zeros(3**n, dtype=bool)
    in_g[visible_index(_keep_visible(enumerate_group(gens)))] = True
    n_bits = circuit.n_mid + n
    in_d = np.zeros((len(ks), 3**n), dtype=bool)
    acc = np.zeros(3**n, dtype=bool)
    done = 0
    for j, k in enumerate(sorted(ks)):
        for size in range(done + 1, min(k, n_bits) + 1):
            combos = itertools.combinations(range(n_bits), size)
            while block := list(itertools.islice(combos, chunk)):
                strings, alive = parity_strings(circuit, block)
                acc[visible_index(_keep_visible(strings[alive]))] = True
        done = max(done, k)
        in_d[j] = acc
    order = np.argsort(np.argsort(ks))
    in_d = in_d[order]
    in_g[0] = False
    in_d[:, 0] = False
    return in_g, in_d


def draw_power(
    n: int, n_targets: int, prior: str, rng: np.random.Generator
) -> np.ndarray:
    """``w_a beta_a^2`` of random dense targets on the ``3^n`` visible strings."""
    if prior not in PRIORS:
        raise ValueError(f"unknown prior {prior!r}")
    weight = (visible_strings(n)).sum(axis=1)
    w = 0.5**weight
    z2 = rng.standard_normal((n_targets, 3**n)) ** 2
    q = z2 * (w if prior == "iid" else 1.0)
    q[:, 0] = 0.0
    return np.asarray(q)


def counting_budget(in_g: np.ndarray, in_d: np.ndarray, q: np.ndarray) -> Budget:
    """Budget from string memberships and the powers ``q_a = w_a beta_a^2``."""
    var = q.sum(axis=1)
    circ = q[:, ~in_g].sum(axis=1) / var
    r2 = np.array([q[:, d].sum(axis=1) / var for d in in_d])
    read = np.array([q[:, in_g & ~d].sum(axis=1) / var for d in in_d])
    return Budget(circuit=circ, readout=read, r2=r2, cross=np.zeros_like(r2))


def clifford_budget(
    circuit: CliffordQELM, ks: tuple[int, ...], q: np.ndarray
) -> Budget:
    """Budget of the Clifford QELM of #337 by counting strings."""
    in_g, in_d = clifford_masks(circuit, ks)
    if np.any(in_d & ~in_g):
        raise AssertionError("a parity string lies outside the effect group")
    return counting_budget(in_g, in_d, q)
