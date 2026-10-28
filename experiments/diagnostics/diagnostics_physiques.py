#!/usr/bin/env python3
"""Check the chaos and the locality of the reservoirs (issue #243).

Runs the diagnostics of :mod:`QELM.evaluation.chaos_locality` on five
reservoirs and draws one figure per diagnostic, each with its data
(:func:`QELM.utils.plots.save_figure`):

- ``tfim_all_to_all``: the TFIM of ``conf/models/TFIM.yaml``, ``H = h sum_i
  Z_i + sum_{i<j} J_ij X_i X_j``, ``J_ij`` uniform in ``[-1/2, 1/2]``, ``h =
  10`` (``--tfim-h``), step ``tau = 0.25``. Symmetry: ``prod_i Z_i``.
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
- ``floquet``: the Floquet Ising chain of Skinner et al. (2019) of
  ``conf/models/floquet_circuit.yaml`` (issue #239); one step is one
  half-layer of bricks, the period two. Symmetry: the reflection (even
  ``N``).

Figures, in ``outputs/diagnostics_physiques/<date_time>/`` (``--out``):

1. ``level_spacing.png`` (part A): histogram of the unfolded spacings of the
   central third of every symmetry sector, pooled over ``--realisations``
   disorder draws, against Poisson, the GOE and the GUE; ``<r>`` in the
   title of each panel.
2. ``entropy_growth.png`` (part B1): half-chain ``S_A(t)`` from the Néel
   state at ``p = 0``, one curve per ``N`` of ``--N-growth``. A local model
   grows linearly at a rate independent of ``N``; the all-to-all one does
   not. The maximal rate ``max_t dS_A/dt`` per ``N`` is in the JSON.
3. ``otoc.png`` (part B2): ``C(x, t) = (1/2) <|[Z_x(t), Z_x0]|^2>`` against
   ``x - x0`` and ``t``, ``x0 = N/2``: a "V" for a local model.
4. ``reference_qubit_delta.png`` (part B3a): ``-<Delta S_Q(x, t)>``, the
   entropy of the reference qubit removed by a measurement at ``(x, t)``,
   at ``--p-volume`` and ``--p-area``.
5. ``reference_qubit_cutoff.png`` (part B3b): ``<S_Q(t)>`` when only the
   outcomes of the measurements at ``|x - x0| <= l`` are kept, at the same
   two ``p``.

The time unit is ``1 / J`` for the Hamiltonians and one half-layer for the
Floquet chain. ``--quick`` runs tiny sizes in a few seconds (tests).

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

from QELM.evaluation.chaos_locality import (
    R_GOE,
    R_GUE,
    R_POISSON,
    brickwork_layers,
    charge_operator,
    entanglement_growth,
    floquet_ising_gate,
    goe_pdf,
    gue_pdf,
    ising_chain_hamiltonian,
    level_statistics,
    neel_state,
    otoc_map,
    parity_operator,
    poisson_pdf,
    propagator_operator,
    quasienergies,
    reference_qubit_cutoff,
    reference_qubit_trajectories,
    reflection_operator,
    sector_spectra,
    step_unitary,
    symmetry_blocks,
    xxz_chain_hamiltonian,
    y_polarized_state,
)
from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.utils import new_sweep_dir
from QELM.utils.plots import COLORS, save_figure, subplots


@dataclass
class Instance:
    """One draw of a reservoir.

    Attributes:
        H: The Hamiltonian, ``None`` for a circuit.
        symmetries: Hermitian operators commuting with ``H`` (or with the
            period of the circuit).
        layers: The unitaries of the half-layers of a circuit, else ``None``.
    """

    H: sp.csr_matrix | None
    symmetries: list[sp.csr_matrix]
    layers: list[np.ndarray] | None = None


@dataclass
class Reservoir:
    """One reservoir and the way to draw an instance of it.

    Attributes:
        key: Short name, used in the JSON.
        label: Title of its panels.
        local: Whether the couplings are nearest-neighbour.
        time_unit: Label of the time axis.
        build: ``build(N, rng)`` draws an :class:`Instance`.
        tau: Time of one step: ``exp(-i H tau)`` for a Hamiltonian, one
            half-layer (``tau = 1``) for a circuit.
    """

    key: str
    label: str
    local: bool
    time_unit: str
    build: Callable[[int, np.random.Generator], Instance]
    tau: float = 1.0

    @property
    def floquet(self) -> bool:
        """Whether the levels are the quasi-energies of a circuit period."""
        return self.key == "floquet"

    def steps(self, instance: Instance) -> list[np.ndarray]:
        """Return the dense unitaries of the time steps, cycled."""
        if instance.layers is not None:
            return instance.layers
        assert instance.H is not None
        return [step_unitary(instance.H, self.tau)]

    def state_steps(self, instance: Instance) -> list[Any]:
        """As :meth:`steps`, applied to a state from the sparse ``H``."""
        if instance.layers is not None:
            return instance.layers
        assert instance.H is not None
        return [propagator_operator(instance.H, self.tau)]


def reservoirs(tfim_h: float = 10.0) -> list[Reservoir]:
    """Return the five reservoirs of the module docstring."""

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

    def floquet(N: int, rng: np.random.Generator) -> Instance:
        layers = brickwork_layers(N, floquet_ising_gate())
        symmetries = [reflection_operator(N)] if N % 2 == 0 else []
        return Instance(None, symmetries, layers)

    return [
        Reservoir(
            "tfim_all_to_all",
            f"TFIM all-to-all ($h={tfim_h:g}$)",
            False,
            "$tJ$",
            tfim,
            tau=0.25,
        ),
        Reservoir("ising_hz0", "Ising 1D $h_z=0$", True, "$tJ$", ising(0.0), 0.3),
        Reservoir("ising_hz05", "Ising 1D $h_z=0.5$", True, "$tJ$", ising(0.5), 0.3),
        Reservoir("xxz", "XXZ 1D ($\\Delta=J$)", True, "$tJ$", xxz, tau=0.1),
        Reservoir("floquet", "Floquet 1D (Skinner)", True, "half-layers", floquet),
    ]


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
        draws = (
            realisations
            if model.key in ("tfim_all_to_all", "ising_hz0", "ising_hz05")
            else 1
        )
        spectra = []
        blocks: list[np.ndarray] | None = None
        for _ in range(draws):
            instance = model.build(N, rng)
            if blocks is None:  # the symmetries do not depend on the disorder
                blocks = symmetry_blocks(2**N, instance.symmetries)
            if instance.layers is not None:
                period = instance.layers[1] @ instance.layers[0]
                spectra.append(quasienergies(period, blocks=blocks))
            else:
                assert instance.H is not None
                spectra.append(sector_spectra(instance.H, blocks=blocks))
        stats = level_statistics(spectra, circular=model.floquet)
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
            steps = model.state_steps(model.build(N, rng))
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
        for i, (label, _) in enumerate(states.values()):
            axes[i, j].set_title(f"{model.label}, {label}")
            axes[i, j].set_xlabel(model.time_unit)
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
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``otoc.png``, ``C(x, t)`` with ``x0 = N // 2``."""
    x0 = N // 2
    fig, axes = subplots(1, len(models), figsize=(4 * len(models), 3.8), squeeze=False)
    summary: dict[str, Any] = {}
    x_edges = np.arange(N + 1) - 0.5 - x0
    for ax, model in zip(axes[0], models, strict=True):
        steps = model.steps(model.build(N, rng))
        C = otoc_map(steps, x0, n_steps)
        t = _times(model, n_steps)
        t_edges = np.concatenate([t - model.tau / 2, [t[-1] + model.tau / 2]])
        mesh = ax.pcolormesh(
            x_edges, t_edges, C, vmin=0, vmax=1.2, cmap="magma", label=model.key
        )
        ax.set_title(model.label)
        ax.set_xlabel("$x - x_0$")
        summary[model.key] = {"front": _front(C, t, x0)}
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
# Part B3


def run_reference_qubit(
    models: Sequence[Reservoir],
    N: int,
    n_steps: int,
    p_values: Sequence[float],
    cutoffs: Sequence[int],
    trajectories: int,
    cutoff_trajectories: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    """Draw ``reference_qubit_delta.png`` and ``reference_qubit_cutoff.png``."""
    x0 = N // 2
    shape = (len(p_values), len(models))
    size = (4 * len(models), 3.4 * len(p_values))
    fig_d, axes_d = subplots(*shape, figsize=size, squeeze=False)
    fig_c, axes_c = subplots(*shape, figsize=size, squeeze=False)
    x_edges = np.arange(N + 1) - 0.5 - x0
    summary: dict[str, Any] = {}
    for j, model in enumerate(models):
        steps = model.steps(model.build(N, rng))
        t = _times(model, n_steps)
        t_edges = np.concatenate([t - model.tau / 2, [t[-1] + model.tau / 2]])
        summary[model.key] = {}
        for i, p in enumerate(p_values):
            result = reference_qubit_trajectories(
                steps, x0, p, n_steps, trajectories, rng
            )
            ax = axes_d[i, j]
            mesh = ax.pcolormesh(
                x_edges,
                t_edges,
                -result.delta_S_Q,
                vmin=0,
                vmax=1,
                cmap="viridis",
                label=f"{model.key} p={p:g}",
            )
            ax.set_title(f"{model.label}, p = {p:g}")
            ax.set_xlabel("$x - x_0$")
            ax.set_ylabel("time" if j == 0 else "")
            S_cut = reference_qubit_cutoff(
                steps, x0, p, n_steps, cutoffs, cutoff_trajectories, rng
            )
            ax = axes_c[i, j]
            ax.plot(
                t,
                result.S_Q,
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
                "S_Q_final": float(result.S_Q[-1]),
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
PARTS = ("spectra", "growth", "otoc", "reference")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfim-h", type=float, default=10.0)
    parser.add_argument("--N-spectra", type=int, default=12)
    parser.add_argument("--realisations", type=int, default=5)
    parser.add_argument("--N-growth", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--growth-steps", type=int, default=40)
    parser.add_argument("--N-otoc", type=int, default=10)
    parser.add_argument("--otoc-steps", type=int, default=24)
    parser.add_argument("--N-ref", type=int, default=8)
    parser.add_argument("--ref-steps", type=int, default=24)
    parser.add_argument("--p-volume", type=float, default=0.05)
    parser.add_argument("--p-area", type=float, default=0.5)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[-1, 0, 1, 2, 4])
    parser.add_argument("--trajectories", type=int, default=400)
    parser.add_argument("--cutoff-trajectories", type=int, default=60)
    parser.add_argument("--models", nargs="+", default=None, help="keys to run")
    parser.add_argument(
        "--parts", nargs="+", choices=PARTS, default=list(PARTS), help="parts to run"
    )
    parser.add_argument(
        "--quick", action="store_true", help="tiny sizes, a few seconds (tests)"
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.N_spectra, args.realisations = 6, 1
        args.N_growth, args.growth_steps = [4, 6], 4
        args.N_otoc, args.otoc_steps = 4, 3
        args.N_ref, args.ref_steps = 4, 3
        args.cutoffs = [-1, 0, 4]
        args.trajectories, args.cutoff_trajectories = 2, 1
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    """Run every diagnostic and return the output folder."""
    matplotlib.use("Agg")
    args = parse_args(argv)
    models = reservoirs(args.tfim_h)
    if args.models:
        models = [m for m in models if m.key in args.models]
    out_dir = args.out or new_sweep_dir("outputs", "diagnostics_physiques")
    out_dir.mkdir(parents=True, exist_ok=True)
    rngs = dict(zip(PARTS, np.random.default_rng(args.seed).spawn(len(PARTS))))
    runs: dict[str, Callable[[], dict[str, Any]]] = {
        "spectra": lambda: run_level_spacing(
            models, args.N_spectra, args.realisations, rngs["spectra"], out_dir
        ),
        "growth": lambda: run_entropy_growth(
            models, args.N_growth, args.growth_steps, rngs["growth"], out_dir
        ),
        "otoc": lambda: run_otoc(
            models, args.N_otoc, args.otoc_steps, rngs["otoc"], out_dir
        ),
        "reference": lambda: run_reference_qubit(
            models,
            args.N_ref,
            args.ref_steps,
            [args.p_volume, args.p_area],
            args.cutoffs,
            args.trajectories,
            args.cutoff_trajectories,
            rngs["reference"],
            out_dir,
        ),
    }
    summary = {part: runs[part]() for part in PARTS if part in args.parts}
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "summary.json").write_text(
        json.dumps({"args": config, **summary}, indent=2) + "\n"
    )
    print(f"Figures written in {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
