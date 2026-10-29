#!/usr/bin/env python3
"""Check the chaos and the locality of the reservoirs (issue #243).

Runs the diagnostics of :mod:`QELM.evaluation.chaos_locality` on two
families of reservoirs and draws, for each family, one figure per
diagnostic, each with its data (:func:`QELM.utils.plots.save_figure`).

**Continuous evolutions** (``continuous/``), ``exp(-i H tau)`` of a
Hamiltonian, time in units of ``1 / J``:

- ``tfim_all_to_all``: the TFIM of ``conf/models/TFIM.yaml``, ``H = h sum_i
  Z_i + sum_{i<j} J_ij X_i X_j``, ``J_ij`` uniform in ``[-1/2, 1/2]``, ``h =
  10`` (``--tfim-h``), ``tau = 0.25``. Symmetry: ``prod_i Z_i``.
- ``ising_hz0``: the open Ising chain ``- sum_i J_i Z_i Z_{i+1} - h_x sum_i
  X_i - h_z sum_i Z_i`` of ``conf/models/continuous_tfim_1d.yaml`` (issue
  #241) with ``h_z = 0``: free fermions, integrable. ``J_i`` uniform in
  ``[0.5, 1.5]``, ``h_x = 1.05``, ``tau = 0.3``. Symmetry: ``prod_i X_i``.
- ``ising_hz05``: the same chain with ``h_z = 0.5`` (Kim and Huse 2013),
  the reservoir of issue #241. No symmetry.
- ``xxz``: the clean Heisenberg chain of ``conf/models/XXZ.yaml`` (issue
  #236), ``J = Delta = 1``, ``h = 0``, ``tau = 0.1`` (its spin waves move
  up to 4 sites per unit time). Symmetries: the charge ``Q``, the reflection
  and, at ``Q = N/2``, the spin flip ``prod_i X_i``.
- ``xxz_disorder``: the same chain with the random fields ``h_i`` uniform in
  ``[-h, h]``, ``h = 1`` (``--xxz-h``), the ``h`` of ``conf/models/XXZ.yaml``.
  Symmetry: the charge ``Q``.

**Discrete evolutions** (``discrete/``), brickwork circuits of two-qubit
gates on an open chain, one step a half-layer (the even bonds, then the odd
ones), time in half-layers:

- ``floquet``: the Floquet Ising chain of Skinner et al. (2019) of
  ``conf/models/floquet_circuit.yaml`` (issue #239), the same brick
  everywhere and at every step. Symmetry: the reflection (even ``N``).
- ``haar``: Haar-random gates of U(4), as ``conf/models/haar_circuit.yaml``.
- ``clifford``: random gates of the two-qubit Clifford group, as
  ``conf/models/clifford_circuit.yaml``.

The Haar and Clifford circuits of the experiments draw new gates at every
step: their dynamics (growth, OTOC, reference qubit) is that random circuit,
averaged over ``--circuit-draws`` draws. Such a circuit has no spectrum; the
level statistics use their Floquet version, one draw of the gates of a
period repeated in time, over ``--realisations`` draws.

Figures, in ``outputs/diagnostics_physiques/<date_time>/<family>/``
(``--out``):

1. ``level_spacing.png`` (part A): histogram of the unfolded spacings of the
   central third of every symmetry sector (the quasi-energies of the period
   for a circuit), pooled over disorder draws, against Poisson, the GOE and
   the GUE; ``<r>`` in the title of each panel.
2. ``entropy_growth.png`` (part B1): half-chain ``S_A(t)`` at ``p = 0`` from
   the Néel state and from ``|+y>^N``, one curve per ``N`` of ``--N-growth``.
   A local model grows linearly at a rate independent of ``N``; the
   all-to-all one does not. The maximal rate per ``N`` is in the JSON.
3. ``otoc.png`` (part B2): ``C(x, t) = (1/2) <|[Z_x(t), Z_x0]|^2>`` against
   ``x - x0`` and ``t``, ``x0 = N/2``: a "V" for a local model.
4. ``reference_qubit_delta.png`` (part B3a): ``-<Delta S_Q(x, t)>``, the
   entropy of the reference qubit removed by a measurement at ``(x, t)``,
   at ``--p-volume`` and ``--p-area``.
5. ``reference_qubit_cutoff.png`` (part B3b): ``<S_Q(t)>`` when only the
   outcomes of the measurements at ``|x - x0| <= l`` are kept, at the same
   two ``p``.

6. ``velocities.png`` (part B2b, the local models): the entanglement
   velocity ``v_E`` (slope of ``S_A(t)`` from random product states), the
   butterfly velocity ``v_B`` (front of ``C(x, t)`` from the edge of the
   chain) and the Lieb-Robinson velocity ``v_LR``, against ``v_E <= v_B <=
   v_LR``.

``--families``, ``--models`` and ``--parts`` run a subset; ``--quick`` runs
tiny sizes in a few seconds (tests).

    uv run python experiments/diagnostics/diagnostics_physiques.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LogNorm

from QELM.evaluation.chaos_locality import (
    R_GOE,
    R_GUE,
    R_POISSON,
    Step,
    brickwork_bonds,
    butterfly_velocity,
    charge_operator,
    dense_operator,
    entanglement_growth,
    entanglement_velocity,
    floquet_ising_gate,
    gate_layer_operator,
    goe_pdf,
    gue_pdf,
    ising_chain_hamiltonian,
    level_statistics,
    lieb_robinson_velocity,
    neel_state,
    operator_spreading,
    otoc_map,
    parity_operator,
    poisson_pdf,
    propagator_operator,
    quasienergies,
    random_brickwork_steps,
    random_floquet_layers,
    random_product_state,
    reference_qubit_cutoff,
    reference_qubit_trajectories,
    reflection_operator,
    sector_spectra,
    step_unitary,
    symmetry_blocks,
    xxz_chain_hamiltonian,
    y_polarized_state,
)
from QELM.models.circuits.clifford import random_clifford_gate
from QELM.models.circuits.haar import haar_gate
from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.utils import new_sweep_dir
from QELM.utils.plots import COLORS, save_figure, subplots


@dataclass
class Instance:
    """One draw of a reservoir, for its level statistics.

    Attributes:
        H: The Hamiltonian, ``None`` for a circuit.
        symmetries: Hermitian operators commuting with ``H`` (or with the
            period of the circuit).
        layers: The even and the odd half-layers of a circuit, else ``None``.
    """

    H: sp.csr_matrix | None
    symmetries: list[sp.csr_matrix]
    layers: list[Step] | None = None


@dataclass
class Reservoir:
    """One reservoir and the way to draw it.

    Attributes:
        key: Short name, used in the JSON.
        label: Title of its panels.
        family: ``"continuous"`` (Hamiltonian) or ``"discrete"`` (circuit).
        time_unit: Label of the time axis.
        build: ``build(N, rng)`` draws an :class:`Instance`.
        tau: Time of one step: ``exp(-i H tau)`` for a Hamiltonian, one
            half-layer (``tau = 1``) for a circuit.
        disordered: Whether two draws differ (averaged over in part A).
        random_gates: For a random circuit, the draw of one gate: its
            dynamics has new gates at every step.
        bond_norm: Largest norm of a two-site term of a local Hamiltonian
            (its Lieb-Robinson velocity), ``None`` for a circuit or a
            non-local model.
        velocity_tau: Time step of the velocity part, finer than ``tau``.
    """

    key: str
    label: str
    family: str
    time_unit: str
    build: Callable[[int, np.random.Generator], Instance]
    tau: float = 1.0
    disordered: bool = False
    random_gates: Callable[[np.random.Generator], np.ndarray] | None = None
    bond_norm: float | None = None
    velocity_tau: float = 1.0

    @property
    def lieb_robinson(self) -> float | None:
        """``v_LR``: 1 site per half-layer for a brickwork circuit, the bound
        of :func:`lieb_robinson_velocity` for a local Hamiltonian, ``None``
        without a light cone."""
        if self.family == "discrete":
            return 1.0
        if self.bond_norm is None:
            return None
        return lieb_robinson_velocity(self.bond_norm)

    def steps(
        self,
        N: int,
        n_steps: int,
        rng: np.random.Generator,
        sparse: bool = False,
        tau: float | None = None,
    ) -> list[Step]:
        """Draw the time steps of the dynamics, cycled by the diagnostics.

        A Hamiltonian gives ``[exp(-i H tau)]``, dense or applied from the
        sparse ``H`` (``sparse=True``), the Floquet chain its two
        half-layers, a random circuit ``n_steps`` half-layers of new gates.
        ``tau`` replaces the time step of a Hamiltonian.
        """
        if self.random_gates is not None:
            return list(random_brickwork_steps(N, n_steps, self.random_gates, rng))
        instance = self.build(N, rng)
        if instance.layers is not None:
            return instance.layers
        assert instance.H is not None
        tau = self.tau if tau is None else tau
        if sparse:
            return [propagator_operator(instance.H, tau)]
        return [step_unitary(instance.H, tau)]


def reservoirs(tfim_h: float = 10.0, xxz_h: float = 1.0) -> list[Reservoir]:
    """Return the reservoirs of the module docstring, both families."""

    def tfim(N: int, rng: np.random.Generator) -> Instance:
        np.random.seed(int(rng.integers(2**31)))
        H = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": tfim_h}).hamiltonian()
        return Instance(H, [parity_operator(N, "Z")])

    def ising(h_z: float) -> Callable[[int, np.random.Generator], Instance]:
        def build(N: int, rng: np.random.Generator) -> Instance:
            couplings = 1.0 + (rng.random(N - 1) - 0.5)  # J = W = 1
            H = ising_chain_hamiltonian(couplings, h_x=1.05, h_z=h_z)
            return Instance(H, [parity_operator(N, "X")] if h_z == 0 else [])

        return build

    def xxz(N: int, rng: np.random.Generator) -> Instance:
        H = xxz_chain_hamiltonian(N, J=1.0, Delta=1.0)
        symmetries = [charge_operator(N), reflection_operator(N)]
        return Instance(H, [*symmetries, parity_operator(N, "X")])

    def xxz_disorder(N: int, rng: np.random.Generator) -> Instance:
        fields = rng.uniform(-xxz_h, xxz_h, size=N)
        H = xxz_chain_hamiltonian(N, J=1.0, Delta=1.0, fields=fields)
        return Instance(H, [charge_operator(N)])

    def floquet(N: int, rng: np.random.Generator) -> Instance:
        gate = floquet_ising_gate()
        layers: list[Step] = [
            gate_layer_operator(N, [(i, j, gate) for i, j in brickwork_bonds(N, h)])
            for h in (0, 1)
        ]
        return Instance(None, [reflection_operator(N)] if N % 2 == 0 else [], layers)

    def random_floquet(
        draw: Callable[[np.random.Generator], np.ndarray],
    ) -> Callable[[int, np.random.Generator], Instance]:
        def build(N: int, rng: np.random.Generator) -> Instance:
            return Instance(None, [], list(random_floquet_layers(N, draw, rng)))

        return build

    hamiltonian = "continuous"
    return [
        Reservoir(
            "tfim_all_to_all",
            f"TFIM all-to-all ($h={tfim_h:g}$)",
            hamiltonian,
            "$tJ$",
            tfim,
            tau=0.25,
            disordered=True,
        ),
        # J_i <= 1.5: bond norm 1.5. XXZ: ||XX + YY + ZZ|| = 3.
        Reservoir(
            "ising_hz0",
            "Ising 1D $h_z=0$",
            hamiltonian,
            "$tJ$",
            ising(0.0),
            0.3,
            True,
            bond_norm=1.5,
            velocity_tau=0.1,
        ),
        Reservoir(
            "ising_hz05",
            "Ising 1D $h_z=0.5$",
            hamiltonian,
            "$tJ$",
            ising(0.5),
            0.3,
            True,
            bond_norm=1.5,
            velocity_tau=0.1,
        ),
        Reservoir(
            "xxz",
            "XXZ 1D clean",
            hamiltonian,
            "$tJ$",
            xxz,
            tau=0.1,
            bond_norm=3.0,
            velocity_tau=0.05,
        ),
        Reservoir(
            "xxz_disorder",
            f"XXZ 1D, $h_i\\in[-{xxz_h:g},{xxz_h:g}]$",
            hamiltonian,
            "$tJ$",
            xxz_disorder,
            tau=0.1,
            disordered=True,
            bond_norm=3.0,
            velocity_tau=0.05,
        ),
        Reservoir("floquet", "Floquet (Skinner)", "discrete", "half-layers", floquet),
        Reservoir(
            "haar",
            "Haar circuit",
            "discrete",
            "half-layers",
            random_floquet(haar_gate),
            disordered=True,
            random_gates=haar_gate,
        ),
        Reservoir(
            "clifford",
            "Clifford circuit",
            "discrete",
            "half-layers",
            random_floquet(random_clifford_gate),
            disordered=True,
            random_gates=random_clifford_gate,
        ),
    ]


FAMILIES = ("continuous", "discrete")


def _times(reservoir: Reservoir, n_steps: int) -> np.ndarray:
    return np.arange(n_steps + 1) * reservoir.tau


# ---------------------------------------------------------------------------
# Part A


def run_level_spacing(
    models: Sequence[Reservoir],
    N: int,
    realisations: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``level_spacing.png`` and return ``<r>`` per model."""
    fig, axes = subplots(1, len(models), figsize=(4 * len(models), 3.6), squeeze=False)
    s_grid = np.linspace(0, 4, 200)
    summary: dict[str, Any] = {}
    for ax, model in zip(axes[0], models, strict=True):
        draws = realisations if model.disordered else 1
        spectra = []
        blocks: list[np.ndarray] | None = None
        for _ in range(draws):
            instance = model.build(N, rng)
            if blocks is None:  # the symmetries do not depend on the disorder
                blocks = symmetry_blocks(2**N, instance.symmetries)
            if instance.layers is not None:
                even, odd = (dense_operator(layer) for layer in instance.layers)
                spectra.append(quasienergies(odd @ even, blocks=blocks))
            else:
                assert instance.H is not None
                spectra.append(sector_spectra(instance.H, blocks=blocks))
        circular = model.family == "discrete"
        stats = level_statistics(spectra, circular=circular)
        if stats.spacings.size:
            ax.hist(
                stats.spacings,
                bins=np.linspace(0, 4, 41),
                density=True,
                color=COLORS[0],
                alpha=0.55,
                label=f"{model.key} spacings",
            )
        ax.plot(s_grid, poisson_pdf(s_grid), color="k", ls="--", label="Poisson")
        ax.plot(s_grid, goe_pdf(s_grid), color=COLORS[1], label="GOE")
        ax.plot(s_grid, gue_pdf(s_grid), color=COLORS[2], ls=":", label="GUE")
        ax.set_title(f"{model.label}\n$\\langle r\\rangle = {stats.mean_r:.3f}$")
        ax.set_xlabel("$s$")
        ax.set_xlim(0, 4)
        summary[model.key] = {
            "mean_r": stats.mean_r,
            "n_spacings": int(stats.spacings.size),
            "sector_dims": stats.sector_dims,
            "realisations": draws,
        }
    axes[0, 0].set_ylabel("$P(s)$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Level spacings, N = {N}: <r> Poisson {R_POISSON:.3f}, "
        f"GOE {R_GOE:.3f}, GUE {R_GUE:.3f}"
    )
    references = {"R_POISSON": R_POISSON, "R_GOE": R_GOE, "R_GUE": R_GUE}
    save_figure(
        fig, out_dir, "level_spacing", {"N": N, "models": summary, **references}
    )
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Part B1


def run_entropy_growth(
    models: Sequence[Reservoir],
    N_values: Sequence[int],
    n_steps: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``entropy_growth.png`` and return the maximal rate per ``N``.

    Two initial product states, one per row: the Néel state ``|0101...>``
    and ``|+y>^N``, whose energy is zero (infinite temperature) for the TFIM
    and the Ising chains, so that ``S_A`` saturates near the Page value.
    """
    states = {"neel": ("Néel", neel_state), "y": ("$|+y\\rangle^N$", y_polarized_state)}
    fig, axes = subplots(
        len(states),
        len(models),
        figsize=(4 * len(models), 3.4 * len(states)),
        squeeze=False,
    )
    summary: dict[str, Any] = {model.key: {} for model in models}
    for j, model in enumerate(models):
        t = _times(model, n_steps)
        for k, N in enumerate(N_values):
            steps = model.steps(N, n_steps, rng, sparse=True)
            for i, (name, (label, state)) in enumerate(states.items()):
                S = entanglement_growth(steps, state(N), n_steps)
                rate = float(np.max(np.diff(S) / np.diff(t)))
                summary[model.key].setdefault(name, {})[str(N)] = {
                    "max_rate": rate,
                    "S_final": float(S[-1]),
                }
                axes[i, j].plot(
                    t,
                    S,
                    color=COLORS[k % len(COLORS)],
                    label=f"{model.key} {name} N={N}",
                )
        for i, (name, (label, _)) in enumerate(states.items()):
            axes[i, j].set_title(f"{model.label}, {label}")
            axes[i, j].set_xlabel(model.time_unit)
            runs = summary[model.key][name].values()
            if max(abs(run["max_rate"]) for run in runs) < 1e-6:
                # |+y>^N is fully polarized: an eigenstate of the SU(2)
                # symmetric Heisenberg chain, so nothing grows.
                axes[i, j].set_ylim(-0.05, 1)
                axes[i, j].text(
                    0.5,
                    0.5,
                    "eigenstate of H:\n$S_A = 0$",
                    ha="center",
                    transform=axes[i, j].transAxes,
                )
    for i in range(len(states)):
        axes[i, 0].set_ylabel("$S_A$ (bits), $A$ = left half")
        axes[i, 0].legend(
            axes[i, 0].get_lines(), [f"N = {N}" for N in N_values], fontsize=8
        )
    fig.suptitle("Entanglement growth from a product state at p = 0")
    save_figure(fig, out_dir, "entropy_growth", {"models": summary})
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Part B2


def run_otoc(
    models: Sequence[Reservoir],
    N: int,
    n_steps: int,
    circuit_draws: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``otoc.png``, ``C(x, t)`` with ``x0 = N // 2``.

    A random circuit is averaged over ``circuit_draws`` draws of its gates.
    """
    x0 = N // 2
    fig, axes = subplots(1, len(models), figsize=(4 * len(models), 3.8), squeeze=False)
    summary: dict[str, Any] = {}
    x_edges = np.arange(N + 1) - 0.5 - x0
    for ax, model in zip(axes[0], models, strict=True):
        draws = circuit_draws if model.random_gates is not None else 1
        C = np.mean(
            [otoc_map(model.steps(N, n_steps, rng), x0, n_steps) for _ in range(draws)],
            axis=0,
        )
        t = _times(model, n_steps)
        t_edges = np.concatenate([t - model.tau / 2, [t[-1] + model.tau / 2]])
        mesh = ax.pcolormesh(
            x_edges, t_edges, C, vmin=0, vmax=1.2, cmap="magma", label=model.key
        )
        ax.set_title(model.label)
        ax.set_xlabel("$x - x_0$")
        summary[model.key] = {"front": _front(C, t, x0), "draws": draws}
    axes[0, 0].set_ylabel("time")
    fig.colorbar(mesh, ax=axes[0, -1], label="$C(x, t)$")
    fig.suptitle(
        f"OTOC $C(x,t)=\\frac{{1}}{{2}}\\langle|[Z_x(t),Z_{{x_0}}]|^2\\rangle$, N = {N}"
    )
    save_figure(fig, out_dir, "otoc", {"N": N, "x0": x0, "models": summary})
    plt.close(fig)
    return summary


def _front(C: np.ndarray, t: np.ndarray, x0: int, threshold: float = 0.1) -> dict:
    """First time at which ``C(x, t) > threshold``, per distance ``|x - x0|``."""
    first: dict[str, float | None] = {}
    for x in range(C.shape[1]):
        above = np.flatnonzero(C[:, x] > threshold)
        first[str(x - x0)] = float(t[above[0]]) if above.size else None
    return first


# ---------------------------------------------------------------------------
# Part B2b


def run_velocities(
    models: Sequence[Reservoir],
    N_otoc: int,
    N_entropy: int,
    n_steps: int,
    draws: int,
    circuit_draws: int,
    thresholds: Sequence[float],
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``velocities.png``: ``v_E``, ``v_B`` and ``v_LR`` of the local models.

    - ``v_B``: front of ``C(x, t) = (1/2) <|[Z_0(t), Z_x]|^2>`` from the edge
      ``x0 = 0`` (:func:`operator_spreading`, ``N_otoc`` sites, distances up
      to ``N_otoc - 1``), at the level ``thresholds[0]``; the other levels
      are in the JSON.
    - ``v_E``: slope of the half-chain ``S_A(t)`` (``N_entropy`` sites) from
      random product states, infinite temperature (``s_eq`` = 1 bit per
      site).
    - ``v_LR``: 1 site per half-layer for a circuit, the Lieb-Robinson bound
      for a Hamiltonian.

    A disordered model is averaged over ``draws`` draws (the OTOC map and
    ``S_A(t)``, one random product state per draw), a random circuit over
    ``circuit_draws`` draws (its single-draw OTOC is noisy: 0 or 2 on every
    site for a Clifford circuit); the clean models use ``draws`` product
    states for ``S_A(t)``.
    """
    local = [m for m in models if m.lieb_robinson is not None]
    summary: dict[str, Any] = {}
    if not local:
        return summary
    fig, axes = subplots(1, 3, figsize=(15, 4.4), squeeze=False)
    ax_front, ax_growth, ax_bars = axes[0]
    for k, model in enumerate(local):
        color = COLORS[k % len(COLORS)]
        tau = model.velocity_tau
        t = np.arange(n_steps + 1) * tau
        model_draws = circuit_draws if model.random_gates is not None else draws
        otoc_draws = model_draws if model.disordered else 1
        C = np.mean(
            [
                operator_spreading(
                    model.steps(N_otoc, n_steps, rng, tau=tau), 0, n_steps
                )
                for _ in range(otoc_draws)
            ],
            axis=0,
        )
        fronts = {th: butterfly_velocity(C, t, 0, th) for th in thresholds}
        S = np.mean(
            [
                entanglement_growth(
                    model.steps(N_entropy, n_steps, rng, sparse=True, tau=tau),
                    random_product_state(N_entropy, rng),
                    n_steps,
                )
                for _ in range(model_draws)
            ],
            axis=0,
        )
        growth = entanglement_velocity(t, S)
        v_B = fronts[thresholds[0]]
        v_LR = model.lieb_robinson
        assert v_LR is not None
        summary[model.key] = {
            "v_E": growth.velocity,
            "v_B": v_B.velocity,
            "v_LR": v_LR,
            "v_B_by_threshold": {str(th): f.velocity for th, f in fronts.items()},
            "front_times": [float(x) for x in v_B.t],
            "front_distances": [float(r) for r in v_B.y],
            "S_max": float(S.max()),
            "otoc_draws": otoc_draws,
            "entropy_draws": model_draws,
        }
        ax_front.plot(v_B.t, v_B.y, "o", color=color, label=model.key)
        ax_front.plot(
            v_B.t,
            v_B.velocity * v_B.t + v_B.intercept,
            "-",
            color=color,
            label=f"{model.label}: $v_B$ = {v_B.velocity:.2f}",
        )
        ax_growth.plot(
            t, S, color=color, label=f"{model.label}: $v_E$ = {growth.velocity:.2f}"
        )
        ax_growth.plot(
            growth.t,
            growth.velocity * growth.t + growth.intercept,
            "--",
            color="black",
            linewidth=1,
            label=f"{model.key} fit",
        )
    time_unit = local[0].time_unit
    ax_front.set_xlabel(f"front time $t^*(r)$ ({time_unit})")
    ax_front.set_ylabel("distance $r$ from $x_0 = 0$ (sites)")
    ax_front.set_title(f"(a) OTOC front, $C(r, t^*) = {thresholds[0]:g}$, N = {N_otoc}")
    ax_front.legend(
        [line for line in ax_front.get_lines() if line.get_linestyle() == "-"],
        [f"{m.label}: $v_B$ = {summary[m.key]['v_B']:.2f}" for m in local],
        fontsize=8,
    )
    ax_growth.set_xlabel(time_unit)
    ax_growth.set_ylabel("$S_A$ (bits), $A$ = left half")
    ax_growth.set_title(f"(b) $S_A(t)$ from random product states, N = {N_entropy}")
    ax_growth.legend(
        [line for line in ax_growth.get_lines() if line.get_linestyle() == "-"],
        [f"{m.label}: $v_E$ = {summary[m.key]['v_E']:.2f}" for m in local],
        fontsize=8,
    )
    x = np.arange(len(local))
    width = 0.27
    for i, (name, label) in enumerate(
        (("v_E", "$v_E$"), ("v_B", "$v_B$"), ("v_LR", "$v_{LR}$"))
    ):
        values = [summary[m.key][name] for m in local]
        bars = ax_bars.bar(
            x + (i - 1) * width, values, width, color=COLORS[i], label=label
        )
        ax_bars.bar_label(bars, fmt="%.2f", fontsize=7)
    ax_bars.set_xticks(x, [m.key for m in local], rotation=20)
    ax_bars.set_yscale("log")
    ax_bars.set_ylabel(f"velocity (sites per {time_unit})")
    ax_bars.set_title("(c) $v_E \\leq v_B \\leq v_{LR}$")
    ax_bars.legend(fontsize=8)
    fig.suptitle("Entanglement, butterfly and Lieb-Robinson velocities")
    save_figure(
        fig,
        out_dir,
        "velocities",
        {
            "N_otoc": N_otoc,
            "N_entropy": N_entropy,
            "n_steps": n_steps,
            "thresholds": list(thresholds),
            "models": summary,
        },
    )
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Part B3


def _share(total: int, parts: int, k: int) -> int:
    """Number of items of part ``k`` when ``total`` is split in ``parts``."""
    return total // parts + (1 if k < total % parts else 0)


def run_reference_qubit(
    models: Sequence[Reservoir],
    N: int,
    n_steps: int,
    p_values: Sequence[float],
    cutoffs: Sequence[int],
    trajectories: int,
    cutoff_trajectories: int,
    circuit_draws: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``reference_qubit_delta.png`` and ``reference_qubit_cutoff.png``.

    The trajectories of a random circuit are split over ``circuit_draws``
    draws of its gates.
    """
    x0 = N // 2
    shape = (len(p_values), len(models))
    size = (4 * len(models), 3.4 * len(p_values))
    fig_d, axes_d = subplots(*shape, figsize=size, squeeze=False)
    fig_c, axes_c = subplots(*shape, figsize=size, squeeze=False)
    x_edges = np.arange(N + 1) - 0.5 - x0
    summary: dict[str, Any] = {}
    for j, model in enumerate(models):
        # A random circuit gets new gates for every batch of trajectories.
        draws = circuit_draws if model.random_gates is not None else 1
        t = _times(model, n_steps)
        t_edges = np.concatenate([t - model.tau / 2, [t[-1] + model.tau / 2]])
        summary[model.key] = {}
        for i, p in enumerate(p_values):
            batches = [
                (
                    model.steps(N, n_steps, rng),
                    _share(trajectories, draws, d),
                    _share(cutoff_trajectories, draws, d),
                )
                for d in range(draws)
            ]
            runs = [
                reference_qubit_trajectories(steps, x0, p, n_steps, n, rng)
                for steps, n, _ in batches
            ]
            weights = np.array([n for _, n, _ in batches], dtype=float)
            S_Q = np.average([r.S_Q for r in runs], axis=0, weights=weights)
            delta = np.average([r.delta_S_Q for r in runs], axis=0, weights=weights)
            ax = axes_d[i, j]
            mesh = ax.pcolormesh(
                x_edges,
                t_edges,
                # Log scale: the light cone is the edge of the faint values,
                # far below the -1 of a measurement of x0 at t = 0. Zeros
                # (outside the cone) are drawn in the lowest color.
                np.clip(-delta, 1e-4, 1),
                norm=LogNorm(vmin=1e-4, vmax=1),
                cmap="viridis",
                label=f"{model.key} p={p:g}",
            )
            ax.set_title(f"{model.label}, p = {p:g}")
            ax.set_xlabel("$x - x_0$")
            ax.set_ylabel("time" if j == 0 else "")
            cut_weights = np.array([n for _, _, n in batches], dtype=float)
            S_cut = np.average(
                [
                    reference_qubit_cutoff(steps, x0, p, n_steps, cutoffs, n, rng)
                    for steps, _, n in batches
                ],
                axis=0,
                weights=cut_weights,
            )
            ax = axes_c[i, j]
            ax.plot(
                t,
                S_Q,
                color="k",
                lw=1,
                ls=":",
                label=f"{model.key} p={p:g} pure",
            )
            for k, cutoff in enumerate(cutoffs):
                name = "none" if cutoff < 0 else f"$\\ell={cutoff}$"
                ax.plot(
                    t,
                    S_cut[k],
                    color=COLORS[k % len(COLORS)],
                    label=f"{model.key} p={p:g} {name}",
                )
            ax.set_title(f"{model.label}, p = {p:g}")
            ax.set_xlabel(model.time_unit)
            ax.set_ylim(-0.02, 1.05)
            summary[model.key][str(p)] = {
                "S_Q_final": float(S_Q[-1]),
                "S_Q_final_cutoff": {
                    str(c): float(S_cut[k, -1]) for k, c in enumerate(cutoffs)
                },
            }
    fig_d.colorbar(mesh, ax=axes_d[:, -1], label="$-\\langle\\Delta S_Q(x,t)\\rangle$")
    fig_d.suptitle(
        f"Reference qubit at $x_0 = {x0}$: entropy removed by a measurement "
        f"at $(x, t)$, N = {N}"
    )
    for ax in axes_c[:, 0]:
        ax.set_ylabel("$\\langle S_Q\\rangle$ (bits)")
    for row in axes_c:
        row[0].legend(
            [*row[0].get_lines()],
            [
                "all outcomes (pure)",
                *["none" if c < 0 else f"|x-x0| <= {c}" for c in cutoffs],
            ],
            fontsize=7,
        )
    fig_c.suptitle(
        f"Reference qubit, outcomes kept only at $|x-x_0|\\leq\\ell$, N = {N}"
    )
    data = {
        "N": N,
        "x0": x0,
        "p_values": list(p_values),
        "cutoffs": list(cutoffs),
        "trajectories": trajectories,
        "cutoff_trajectories": cutoff_trajectories,
        "models": summary,
    }
    save_figure(fig_d, out_dir, "reference_qubit_delta", data)
    save_figure(fig_c, out_dir, "reference_qubit_cutoff", data)
    plt.close(fig_d)
    plt.close(fig_c)
    return summary


# ---------------------------------------------------------------------------


# Each part draws from its own generator, spawned from --seed, so that a part
# run alone (--parts) gives the same figure as in the full run.
PARTS = ("spectra", "growth", "otoc", "reference", "velocities")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfim-h", type=float, default=10.0)
    parser.add_argument("--xxz-h", type=float, default=1.0)
    parser.add_argument("--N-spectra", type=int, default=12, help="continuous")
    parser.add_argument("--N-spectra-discrete", type=int, default=10)
    parser.add_argument("--realisations", type=int, default=5)
    parser.add_argument("--circuit-draws", type=int, default=4)
    parser.add_argument("--N-growth", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--growth-steps", type=int, default=40)
    parser.add_argument("--N-otoc", type=int, default=10)
    parser.add_argument("--otoc-steps", type=int, default=24)
    parser.add_argument("--N-velocity-otoc", type=int, default=11)
    parser.add_argument("--N-velocity-entropy", type=int, default=12)
    parser.add_argument("--velocity-steps", type=int, default=80, help="continuous")
    parser.add_argument("--velocity-steps-discrete", type=int, default=50)
    parser.add_argument("--velocity-draws", type=int, default=6)
    parser.add_argument("--velocity-circuit-draws", type=int, default=16)
    parser.add_argument("--front-thresholds", type=float, nargs="+", default=[0.5, 0.2])
    parser.add_argument("--N-ref", type=int, default=8)
    parser.add_argument("--ref-steps", type=int, default=24)
    parser.add_argument("--p-volume", type=float, default=0.05)
    parser.add_argument("--p-area", type=float, default=0.5)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[-1, 0, 1, 2, 4])
    parser.add_argument("--trajectories", type=int, default=400)
    parser.add_argument("--cutoff-trajectories", type=int, default=60)
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--models", nargs="+", default=None, help="keys to run")
    parser.add_argument(
        "--parts", nargs="+", choices=PARTS, default=list(PARTS), help="parts to run"
    )
    parser.add_argument(
        "--quick", action="store_true", help="tiny sizes, a few seconds (tests)"
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.N_spectra, args.N_spectra_discrete, args.realisations = 6, 6, 1
        args.circuit_draws = 2
        args.N_growth, args.growth_steps = [4, 6], 4
        args.N_otoc, args.otoc_steps = 4, 3
        args.N_ref, args.ref_steps = 4, 3
        args.N_velocity_otoc, args.N_velocity_entropy = 6, 6
        args.velocity_steps, args.velocity_steps_discrete = 12, 12
        args.velocity_draws, args.velocity_circuit_draws = 2, 2
        args.cutoffs = [-1, 0, 4]
        args.trajectories, args.cutoff_trajectories = 2, 2
    return args


def run_family(
    family: str, models: Sequence[Reservoir], args: argparse.Namespace, out_dir: Path
) -> dict[str, Any]:
    """Run the parts of ``args.parts`` on one family, in ``out_dir / family``."""
    folder = out_dir / family
    seed = [args.seed, FAMILIES.index(family)]
    rngs = dict(zip(PARTS, np.random.default_rng(seed).spawn(len(PARTS))))
    N_spectra = args.N_spectra if family == "continuous" else args.N_spectra_discrete
    velocity_steps = (
        args.velocity_steps if family == "continuous" else args.velocity_steps_discrete
    )
    runs: dict[str, Callable[[], dict[str, Any]]] = {
        "spectra": lambda: run_level_spacing(
            models, N_spectra, args.realisations, rngs["spectra"], folder
        ),
        "growth": lambda: run_entropy_growth(
            models, args.N_growth, args.growth_steps, rngs["growth"], folder
        ),
        "otoc": lambda: run_otoc(
            models,
            args.N_otoc,
            args.otoc_steps,
            args.circuit_draws,
            rngs["otoc"],
            folder,
        ),
        "reference": lambda: run_reference_qubit(
            models,
            args.N_ref,
            args.ref_steps,
            [args.p_volume, args.p_area],
            args.cutoffs,
            args.trajectories,
            args.cutoff_trajectories,
            args.circuit_draws,
            rngs["reference"],
            folder,
        ),
        "velocities": lambda: run_velocities(
            models,
            args.N_velocity_otoc,
            args.N_velocity_entropy,
            velocity_steps,
            args.velocity_draws,
            args.velocity_circuit_draws,
            args.front_thresholds,
            rngs["velocities"],
            folder,
        ),
    }
    return {part: runs[part]() for part in PARTS if part in args.parts}


def main(argv: Sequence[str] | None = None) -> Path:
    """Run every diagnostic on every family and return the output folder."""
    matplotlib.use("Agg")
    args = parse_args(argv)
    models = reservoirs(args.tfim_h, args.xxz_h)
    if args.models:
        models = [m for m in models if m.key in args.models]
    out_dir = args.out or new_sweep_dir("outputs", "diagnostics_physiques")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for family in args.families:
        members = [m for m in models if m.family == family]
        if members:
            summary[family] = run_family(family, members, args, out_dir)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "summary.json").write_text(
        json.dumps({"args": config, **summary}, indent=2) + "\n"
    )
    print(f"Figures written in {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
