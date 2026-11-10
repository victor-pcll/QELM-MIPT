"""Figures of the operator light cones, OTOC and Krylov (issue #315).

Reads the JSON files of ``operator_cones.py`` in ``--data`` and writes, in
``--out`` (default: the same folder):

- ``otoc_chain_cones.png``: outcome-averaged Pauli-weight density, N = 41;
- ``otoc_trajectory_cones.png``: trajectory-resolved Pauli weight of the
  connected image of the reference qubit, N = 10, and N = 8 in the summary;
- ``otoc_trajectory_purification.png``: entropy of the reference qubit,
  weight and normalized profiles of the trajectories;
- ``krylov_cones.png``: Krylov wave function ``|phi_n(t)|^2`` of the Floquet
  channel, N = 8, and the saturated ``K`` against p;
- ``krylov_coefficients.png``: ``h_{n+1,n}``, ``K(t)`` and the Lanczos
  ``b_n`` of Parker et al. for the Ising chain.

    uv run python experiments/operator_cones/plot_operator_cones.py \
        --data outputs/operator_cones
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from QELM.evaluation.operator_cones import cone_width  # noqa: E402
from QELM.utils.plots.style import COLORS, subplots  # noqa: E402

P_MAIN = [0.0, 0.05, 0.17, 0.35, 0.65]
RHO_SAT = 0.75  # Pauli-weight density of a scrambled site, 3/4
MPS_TOL = 0.01  # discarded weight above which an MPS row is hatched
THRESHOLD = RHO_SAT / 2
V_B = 0.6  # (q^2 - 1) / (q^2 + 1), q = 2: butterfly velocity of Haar U(4)
P_C = 0.17  # MIPT of the Haar qubit brickwork (Zabalo et al., PRB 101, 060301)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _runs(data: dict) -> list[dict]:
    return sorted(data["runs"].values(), key=lambda r: r["p"])


def _run_at(data: dict, p: float) -> dict:
    return min(data["runs"].values(), key=lambda r: abs(r["p"] - p))


def _cone_axes(ax: plt.Axes, x0: int, N: int, T: int, v_b: bool = True) -> None:
    t = np.linspace(0, T, 50)
    for s in (1, -1):
        ax.plot(s * t, t, color="cyan", ls="--", lw=1.2)
        if v_b:
            ax.plot(s * V_B * t, t, color="white", ls=":", lw=1.2)
    ax.set_xlim(-x0 - 0.5, N - 1 - x0 + 0.5)
    ax.set_ylim(-0.5, T + 0.5)
    ax.set_xlabel(r"$x - x_0$")
    ax.set_ylabel("t (half-layers)")


def _heat(
    ax: plt.Axes, D: np.ndarray, x0: int, vmax: float = RHO_SAT
) -> plt.cm.ScalarMappable:
    T1, N = D.shape
    return ax.imshow(
        np.nan_to_num(D, nan=0.0),
        origin="lower",
        aspect="auto",
        extent=(-x0 - 0.5, N - 1 - x0 + 0.5, -0.5, T1 - 0.5),
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )


def plot_chain(data: dict, out: Path, t_star: int = 20) -> None:
    N, T, x0 = data["N"], data["T"], data["x0"]
    t_star = min(t_star, T)
    fig, axes = subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    im = None
    for ax, p in zip(axes[:5], P_MAIN):
        run = _run_at(data, p)
        im = _heat(ax, np.asarray(run["density"]), x0)
        _cone_axes(ax, x0, N, T)
        ax.set_title(f"p = {run['p']:g}")
        ax.axhline(t_star, color="0.7", lw=0.8)
    assert im is not None
    fig.colorbar(
        im, ax=axes[:5].tolist(), shrink=0.6, label=r"$\rho(x,t)$, surviving operator"
    )
    ax = axes[5]
    runs = _runs(data)
    p = np.array([r["p"] for r in runs])
    width = [cone_width(np.asarray(r["density"])[t_star], THRESHOLD) for r in runs]
    size = [r["size"][t_star] for r in runs]
    ax.plot(p, width, "o-", color=COLORS[0], label=r"width, $\rho \geq 3/8$")
    fwhm = [
        cone_width(
            np.asarray(r["density"])[t_star], np.nanmax(r["density"][t_star]) / 2
        )
        for r in runs
    ]
    ax.plot(p, fwhm, "d-", color=COLORS[3], label=r"width, $\rho \geq \max/2$")
    ax.plot(p, size, "s-", color=COLORS[1], label=r"size $\sum_x \rho$")
    ax.axvline(P_C, color="0.5", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel(f"sites at t = {t_star}")
    ax2 = ax.twinx()
    ax2.semilogy(p, [r["survival"][t_star] for r in runs], "^:", color=COLORS[2])
    ax2.set_ylabel(r"surviving norm $\sum_P E|c_P|^2$", color=COLORS[2])
    ax.legend(loc="upper right")
    ax.set_title("summary")
    fig.suptitle(
        f"Outcome-averaged Pauli weight of $Z_{{x_0}}(t)$, Haar U(4) brickwork, "
        f"N = {N} (exact Haar average, {data['population']} strings)"
    )
    fig.savefig(out / "otoc_chain_cones.png", bbox_inches="tight")
    plt.close(fig)


def _t_star(data: dict) -> int:
    return int(data["N"]) // 2


def plot_trajectories(main: dict, others: list[dict], out: Path) -> None:
    N, T, x0 = main["N"], main["T"], main["x0"]
    fig, axes = subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    im = None
    for ax, p in zip(axes[:5], P_MAIN):
        run = _run_at(main, p)
        im = _heat(ax, np.asarray(run["weighted_density"]), x0)
        _cone_axes(ax, x0, N, T)
        ax.set_title(f"p = {run['p']:g}  ({run['n_traj']} traj.)")
        ax.axhline(_t_star(main), color="0.7", lw=0.8)
    assert im is not None
    fig.colorbar(
        im,
        ax=axes[:5].tolist(),
        shrink=0.6,
        label=r"$E[w\,\rho(x,t)]$, quantum image of R",
    )
    ax = axes[5]
    ax2 = ax.twinx()
    for k, data in enumerate(sorted([main] + others, key=lambda d: d["N"])):
        runs = _runs(data)
        p = np.array([r["p"] for r in runs])
        ts = _t_star(data)
        width = [
            cone_width(np.asarray(r["weighted_density"])[ts], THRESHOLD) / data["N"]
            for r in runs
        ]
        ax.plot(
            p,
            width,
            "o-",
            color=COLORS[k],
            label=f"width / N, N = {data['N']}, t = N/2",
        )
        s_sys = [r["entropy_sys"][-1] / data["N"] for r in runs]
        ax2.plot(p, s_sys, "s--", color=COLORS[k], alpha=0.7)
    ax.axvline(P_C, color="0.5", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel(r"cone width / N ($E[w\rho] \geq 3/8$)")
    ax2.set_ylabel(r"$E[S_{sys}]/N$ at t = 1.5 N (dashed)")
    ax.legend(loc="upper right")
    ax.set_title("summary")
    fig.suptitle(
        f"Trajectory-resolved Pauli weight of the image of a reference qubit, "
        f"monitored Haar U(4), N = {N} (exact)"
    )
    fig.savefig(out / "otoc_trajectory_cones.png", bbox_inches="tight")
    plt.close(fig)


def plot_purification(main: dict, others: list[dict], out: Path) -> None:
    fig, axes = subplots(1, 4, figsize=(23, 5.2))
    T = main["T"]
    t = np.arange(T + 1)
    for k, p in enumerate(P_MAIN):
        run = _run_at(main, p)
        axes[0].errorbar(
            t,
            np.asarray(run["entropy_sys"]) / main["N"],
            yerr=np.asarray(run["entropy_sys_err"]) / main["N"],
            color=COLORS[k],
            label=f"p = {run['p']:g}",
        )
        axes[2].semilogy(t, np.maximum(run["weight"], 1e-12), color=COLORS[k])
        D = np.asarray(run["weighted_density"])[_t_star(main)]
        w = run["weight"][_t_star(main)]
        if w > 1e-9:
            x = np.arange(main["N"]) - main["x0"]
            axes[3].plot(x, D / w, "o-", color=COLORS[k])
    for k, data in enumerate(sorted([main] + others, key=lambda d: d["N"])):
        runs = _runs(data)
        axes[1].errorbar(
            [r["p"] for r in runs],
            [r["entropy_sys"][-1] / data["N"] for r in runs],
            yerr=[r["entropy_sys_err"][-1] / data["N"] for r in runs],
            fmt="o-",
            color=COLORS[k],
            label=f"N = {data['N']}, t = {data['T']}",
        )
    axes[1].axvline(P_C, color="0.5", ls="--", lw=1)
    axes[0].set_xlabel("t (half-layers)")
    axes[0].set_ylabel(r"$E[S_{sys}] / N$")
    axes[0].legend()
    axes[0].set_title(f"purification of the system, N = {main['N']}")
    axes[1].set_xlabel("p")
    axes[1].set_ylabel(r"$E[S_{sys}] / N$ at t = 1.5 N")
    axes[1].set_title("purification against p")
    axes[1].legend()
    axes[2].set_xlabel("t (half-layers)")
    axes[2].set_ylabel(r"$E[w]$")
    axes[2].set_title("quantum weight of the image of R")
    axes[3].set_xlabel(r"$x - x_0$")
    axes[3].set_ylabel(r"$E[w\rho] / E[w]$")
    axes[3].set_title(f"normalized profile at t = {_t_star(main)}")
    axes[3].axhline(RHO_SAT, color="0.6", ls=":")
    fig.tight_layout()
    fig.savefig(out / "otoc_trajectory_purification.png", bbox_inches="tight")
    plt.close(fig)


def plot_krylov(main: dict, others: list[dict], out: Path) -> None:
    n_max = main["n_max"]
    fig, axes = subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    im = None
    for ax, p in zip(axes[:5], P_MAIN):
        run = _run_at(main, p)
        P = np.asarray(run["prob"])
        im = ax.imshow(
            np.log10(np.maximum(P, 1e-6)),
            origin="lower",
            aspect="auto",
            extent=(-0.5, n_max - 0.5, -0.5, n_max - 0.5),
            cmap="magma",
            vmin=-4,
            vmax=0,
            interpolation="nearest",
        )
        t = np.arange(n_max)
        ax.plot(t, t, color="cyan", ls="--", lw=1.2)
        ax.plot(run["complexity"], t, color="white", lw=1.5)
        ax.set_xlim(-0.5, n_max - 0.5)
        ax.set_ylim(-0.5, n_max - 0.5)
        ax.set_xlabel("Krylov site n")
        ax.set_ylabel("t (periods)")
        ax.set_title(f"p = {run['p']:g}")
    assert im is not None
    fig.colorbar(
        im, ax=axes[:5].tolist(), shrink=0.6, label=r"$\log_{10}|\phi_n(t)|^2$"
    )
    ax = axes[5]
    ax2 = ax.twinx()
    for k, data in enumerate(sorted([main] + others, key=lambda d: d["N"])):
        runs = _runs(data)
        p = np.array([r["p"] for r in runs])
        ax.errorbar(
            p,
            [r["complexity"][-1] for r in runs],
            yerr=[r["complexity_err"][-1] for r in runs],
            fmt="o-",
            color=COLORS[k],
            label=f"K(t = {data['n_max'] - 1}), N = {data['N']}",
        )
        lam = [np.mean(r["slowest_modulus"]) for r in runs]
        ax2.plot(p, -np.log(lam), "s--", color=COLORS[k], alpha=0.7)
    ax.axvline(P_C, color="0.5", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("saturated Krylov complexity")
    ax2.set_ylabel(r"decay rate $-\ln|\lambda_1|$ (dashed)")
    ax.legend(loc="upper right")
    ax.set_title("summary")
    fig.suptitle(
        f"Krylov chain of the outcome-averaged monitored Floquet Haar circuit, "
        f"N = {main['N']}, {main['seeds']} circuits; dashed n = t, white K(t)"
    )
    fig.savefig(out / "krylov_cones.png", bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(main: dict, parker: dict | None, out: Path) -> None:
    fig, axes = subplots(1, 3, figsize=(18, 5.2))
    n = np.arange(1, main["n_max"])
    t = np.arange(main["n_max"])
    for k, p in enumerate(P_MAIN):
        run = _run_at(main, p)
        axes[0].plot(n, run["subdiagonal"], color=COLORS[k], label=f"p = {run['p']:g}")
        axes[1].plot(t, run["complexity"], color=COLORS[k])
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"$|h_{n+1,n}|$")
    axes[0].set_title(f"Arnoldi hoppings, N = {main['N']}")
    axes[0].legend()
    axes[1].plot(t, t, color="0.5", ls="--", label="K = t (bound)")
    axes[1].set_xlabel("t (periods)")
    axes[1].set_ylabel("K(t)")
    axes[1].set_title("Krylov complexity")
    axes[1].legend()
    if parker is not None:
        for k, (name, m) in enumerate(sorted(parker["models"].items())):
            b = np.asarray(m["b"])
            ls = "-" if name.startswith("chaotic") else "--"
            axes[2].plot(
                np.arange(1, b.size + 1),
                b,
                ls,
                color=COLORS[k],
                label=name.replace("_", ", "),
            )
        axes[2].set_xlabel("n")
        axes[2].set_ylabel(r"$b_n$")
        axes[2].set_title(r"Parker et al.: Ising chain, $O = X_{N/2}$")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "krylov_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def plot_single(main: dict, others: list[dict], out: Path) -> None:
    """One operator in the circuits of the effect rank (PR #314)."""
    N, T, x0 = main["N"], main["T"], main["x0"]
    S = 2 * T
    fig, axes = subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    im = None
    for ax, p in zip(axes[:5], P_MAIN):
        run = _run_at(main, p)
        im = _heat(ax, np.asarray(run["density"]), x0)
        ax.set_xlim(-x0 - 0.5, N - 1 - x0 + 0.5)
        ax.set_ylim(-0.5, S + 0.5)
        ax.set_xlabel(r"$x - x_0$")
        ax.set_ylabel("half-layers before the final readout")
        ax.set_title(f"p = {run['p']:g}")
        top = ax.twiny()
        top.plot(run["log2_dim_mean"], np.arange(S + 1), color="cyan", lw=2)
        top.set_xlim(0, 2 * N)
        top.set_xlabel(r"$\log_2$ dim of the span (cyan)", color="cyan")
    assert im is not None
    fig.colorbar(im, ax=axes[:5].tolist(), shrink=0.6, label=r"$\rho(x)$ of the span")
    ax = axes[5]
    for k, data in enumerate(sorted([main] + others, key=lambda d: d["N"])):
        runs = _runs(data)
        p = np.array([r["p"] for r in runs])
        ax.plot(
            p,
            [r["log2_dim_mean"][-1] for r in runs],
            "o-",
            color=COLORS[k],
            label=f"one operator, N = {data['N']}",
        )
        if runs[0]["effect_rank"]:
            ax.plot(
                p,
                [np.mean(np.log2(r["effect_rank"])) for r in runs],
                "s--",
                color=COLORS[k],
                alpha=0.7,
                label=f"effect rank #314, N = {data['N']}",
            )
    ax.axvline(P_C, color="0.5", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel(r"$\langle \log_2 \dim \rangle$ after T = N layers")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("summary")
    fig.suptitle(
        f"One operator $Z_{{x_0}}$ in the circuits and masks of PR #314, Haar U(4), "
        f"N = {N}, T = N, {main['seeds']} circuits"
    )
    fig.savefig(out / "single_operator_cones.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = subplots(1, 3, figsize=(18, 5.2))
    s_ax = np.arange(S + 1)
    for k, p in enumerate(P_MAIN):
        run = _run_at(main, p)
        axes[0].plot(
            s_ax, run["log2_dim_mean"], color=COLORS[k], label=f"p = {run['p']:g}"
        )
        axes[1].semilogy(s_ax, np.maximum(run["averaged_norm"], 1e-16), color=COLORS[k])
        D = np.asarray(run["averaged_density"])[-1]
        axes[2].plot(np.arange(N) - x0, D, "o-", color=COLORS[k])
    axes[0].set_xlabel("half-layers before the final readout")
    axes[0].set_ylabel(r"$\langle \log_2 \dim \rangle$")
    axes[0].set_title(f"record-resolved: the span grows, N = {N}")
    axes[0].legend()
    axes[1].set_xlabel("half-layers before the final readout")
    axes[1].set_ylabel(r"$\|Z_{x_0}\|^2$ under the averaged channel")
    axes[1].set_title("outcome-averaged: one operator, norm lost")
    axes[2].set_xlabel(r"$x - x_0$")
    axes[2].set_ylabel(r"$\rho(x)$, averaged channel, input time")
    axes[2].axhline(RHO_SAT, color="0.6", ls=":")
    axes[2].set_title("averaged operator at the input")
    fig.tight_layout()
    fig.savefig(out / "single_operator_growth.png", bbox_inches="tight")
    plt.close(fig)


def plot_branch(main: dict, others: list[dict], out: Path) -> None:
    """One sampled branch of ``Z_x0`` per record, circuits of PR #314."""
    N, T, x0 = main["N"], main["T"], main["x0"]
    S = 2 * T
    fig, axes = subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    im = None
    s_ax = np.arange(S + 1)
    for ax, p in zip(axes[:5], P_MAIN):
        run = _run_at(main, p)
        im = _heat(ax, np.asarray(run["density"]), x0)
        for sign in (-1, 1):
            ax.plot(sign * s_ax, s_ax, color="cyan", ls="--", lw=1.2)
        if run.get("discarded") is not None:
            bad = np.flatnonzero(np.asarray(run["discarded"]) > MPS_TOL)
            if bad.size:
                ax.axhspan(
                    bad[0] - 0.5, S + 0.5, fc="none", ec="0.85", hatch="//", lw=0
                )
                ax.axhline(bad[0] - 0.5, color="white", lw=1.5)
        ax.set_xlim(-x0 - 0.5, N - 1 - x0 + 0.5)
        ax.set_ylim(-0.5, S + 0.5)
        ax.set_xlabel(r"$x - x_0$")
        ax.set_ylabel("half-layers before the final readout")
        ax.set_title(f"p = {run['p']:g}")
    assert im is not None
    fig.colorbar(im, ax=axes[:5].tolist(), shrink=0.6, label=r"$\rho(x)$ of one branch")
    ax = axes[5]
    for k, data in enumerate(sorted([main] + others, key=lambda d: d["N"])):
        runs = [r for r in _runs(data) if any(np.isclose(r["p"], q) for q in P_MAIN)]
        p = [r["p"] for r in runs]
        # Second time: N / 2 half-layers, at most 8 (MPS rows are exact up to 8).
        for s, ls, mk in ((2, "--", "s"), (min(data["N"] // 2, 8), "-", "o")):
            ax.errorbar(
                p,
                [r["size"][s] / data["N"] for r in runs],
                yerr=[r["size_err"][s] / data["N"] for r in runs],
                fmt=mk + ls,
                color=COLORS[k],
                label=f"N = {data['N']}, {s} half-layers",
            )
    ax.axvline(P_C, color="0.5", ls="--", lw=1)
    ax.axhline(RHO_SAT, color="0.6", ls=":")
    ax.set_xlabel("p")
    ax.set_ylabel(r"size $\sum_x \rho / N$ of one branch")
    ax.legend(fontsize=9)
    ax.set_title("summary")
    mps = f", MPS bond {main['chi']}" if main.get("chi") else ""
    fig.suptitle(
        f"One sampled branch of $Z_{{x_0}}$ per record, circuits and masks of PR #314, "
        f"Haar U(4), N = {N}, {main['seeds']} circuits x {main['branches']} records"
        + mps
    )
    fig.savefig(out / f"single_operator_branch_N{N}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("outputs/operator_cones"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.data
    out.mkdir(parents=True, exist_ok=True)
    chain = _load(args.data / "chain.json")
    if chain:
        plot_chain(chain, out)
    traj = sorted(
        (d for d in (_load(f) for f in args.data.glob("traj_N*.json")) if d),
        key=lambda d: d["N"],
    )
    if traj:
        plot_trajectories(traj[-1], traj[:-1], out)
        plot_purification(traj[-1], traj[:-1], out)
    kry = sorted(
        (d for d in (_load(f) for f in args.data.glob("krylov_N*.json")) if d),
        key=lambda d: d["N"],
    )
    if kry:
        plot_krylov(kry[-1], kry[:-1], out)
        plot_coefficients(kry[-1], _load(args.data / "parker.json"), out)
    single = sorted(
        (d for d in (_load(f) for f in args.data.glob("single_N*.json")) if d),
        key=lambda d: d["N"],
    )
    full = [d for d in single if all(f"{p:g}" in d["runs"] for p in P_MAIN)]
    if full:
        plot_single(full[-1], full[:-1], out)
    branch = sorted(
        (d for d in (_load(f) for f in args.data.glob("branch_N*.json")) if d),
        key=lambda d: d["N"],
    )
    for data in branch:
        plot_branch(data, [d for d in branch if d is not data], out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
