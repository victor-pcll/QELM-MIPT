#!/usr/bin/env python3
"""Decisive figures: can the QELM (ridge on M, <P_k>, <P_k P_l>) see the MIPT?

Issue #303. No simulation: every curve is read from results already on
``main`` or in the project folder, and the script applies the same
finite-size criterion to every quantity:

- crossings ``p*(N, N')`` of the curves of consecutive sizes, with a Monte
  Carlo error (each point redrawn from a Gaussian of its standard error,
  2000 draws) and the share of draws that have a crossing in the window;
- the normalized skill ``g = (acc - 1/2) / (acc_Bayes - 1/2)``, the share of
  the Bayes advantage that a decoder recovers on the same trajectories;
- the shrinkage test of the pooled readouts: ``yhat - ybar`` against
  ``y - ybar``.

Sources (read only):

- ``experiments/dehghani_decoder/figures/summary.json`` (#287, PR #289):
  Clifford brickwork, fixed circuits, ``T = 2N``, ridge and MLP on ``M``,
  the full QELM ``M + <P_k> + <P_i P_j>`` and the exact Bayes decoder;
- ``sources/clifford_state_readout_*_order_parameter.json`` (#292, PR #295):
  oracle ``<S_R>`` and pooled readouts of the final Clifford state;
- ``sources/charge_variance_order_parameter.json`` (#299, PR #300): U(1)
  circuits, oracle ``4 <Var Q>`` and pooled linear / quadratic readouts.

Writes into ``--out`` (default ``experiments/qelm_verdict/figures``):
``fig1_meme_circuit.png``, ``fig2_controles_u1.png``,
``fig3_retrecissement.png``, ``fig4_critere_fss.png`` and
``verdict_summary.json`` (every crossing and fit).

Run from the repository root:

    uv run python experiments/qelm_verdict/plot_verdict.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEHGHANI = ROOT / "experiments" / "dehghani_decoder" / "figures" / "summary.json"
CLIFF_SCR = HERE / "sources" / "clifford_state_readout_scrambled_order_parameter.json"
CLIFF_T2N = HERE / "sources" / "clifford_state_readout_T2N_order_parameter.json"
CHARGE = HERE / "sources" / "charge_variance_order_parameter.json"

P_C_CLIFFORD = 0.16  # Gullans & Huse 2020, half-layer steps (reproduced in #292)
P_SHARP = 0.09  # charge sharpening, Barratt et al. 2022 (order of magnitude)
CMAP = plt.get_cmap("viridis")


def colors(n: int) -> list:
    return [CMAP(0.1 + 0.75 * i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------- crossings


def sign_changes(p: np.ndarray, d: np.ndarray, lo: float, hi: float) -> list[float]:
    """Linear-interpolated zeros of ``d`` between grid points inside [lo, hi]."""
    out = []
    for k in range(len(p) - 1):
        if p[k] < lo or p[k + 1] > hi:
            continue
        if d[k] == 0.0:
            out.append(float(p[k]))
        elif d[k] * d[k + 1] < 0:
            out.append(float(p[k] - d[k] * (p[k + 1] - p[k]) / (d[k + 1] - d[k])))
    return out


def crossing(p, ya, ea, yb, eb, lo=0.03, hi=0.6, n_mc=2000, seed=0) -> dict:
    """Crossings of two curves, on the means and over Gaussian redraws."""
    p, ya, ea, yb, eb = (np.asarray(v, dtype=float) for v in (p, ya, ea, yb, eb))
    mean = sign_changes(p, ya - yb, lo, hi)
    rng = np.random.default_rng(seed)
    firsts, counts = [], []
    for _ in range(n_mc):
        da = ya + ea * rng.standard_normal(len(p))
        db = yb + eb * rng.standard_normal(len(p))
        z = sign_changes(p, da - db, lo, hi)
        counts.append(len(z))
        if z:
            firsts.append(
                z[np.argmin([abs(x - (mean[0] if mean else z[0])) for x in z])]
            )
    firsts = np.asarray(firsts)
    # resolved crossing: the difference is significantly negative (z < -2) at
    # some p and significantly positive (z > +2) at a larger p, or the reverse
    sel = (p >= lo) & (p <= hi)
    z = (ya - yb)[sel] / np.sqrt(ea[sel] ** 2 + eb[sel] ** 2 + 1e-30)
    ps = p[sel]
    resolved = None
    neg, pos = np.where(z < -2)[0], np.where(z > 2)[0]
    for first, second in ((neg, pos), (pos, neg)):
        if len(first) and len(second) and first.min() < second.max():
            i = first.min()
            j = second[second > i].min()
            # last significant point of the first sign before j
            i = first[first < j].max()
            inside = [x for x in mean if ps[i] <= x <= ps[j]]
            resolved = {
                "p_low": float(ps[i]),
                "p_high": float(ps[j]),
                "p_star": float(np.median(inside))
                if inside
                else float(0.5 * (ps[i] + ps[j])),
            }
            break
    return {
        "resolved": resolved,
        "mean_curve_crossings": mean,
        "share_with_crossing": float(np.mean(np.asarray(counts) > 0)),
        "share_with_one_crossing": float(np.mean(np.asarray(counts) == 1)),
        "p_star": float(np.median(firsts)) if len(firsts) else None,
        "p_star_err": float(np.std(firsts)) if len(firsts) > 1 else None,
    }


def consecutive_crossings(p, curves: dict[int, tuple], **kw) -> dict:
    sizes = sorted(curves)
    out = {}
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        ya, ea = curves[a]
        yb, eb = curves[b]
        out[f"{a}-{b}"] = crossing(p, ya, ea, yb, eb, **kw)
    return out


# ---------------------------------------------------------------- loaders


def load_dehghani() -> dict:
    d = json.loads(DEHGHANI.read_text())
    out: dict = {"p": np.asarray(d["N6"]["M"]["p"]), "case1": {}, "case3": {}}
    for key, block in d.items():
        n = int(key[1:])
        m = block["M"]
        out["case1"][n] = {
            k: (np.asarray(m[k]["mean"], float), np.asarray(m[k]["err"], float))
            for k in (
                "purified",
                "ridge_acc_all",
                "ridge_acc_pure",
                "mlp_acc_all",
                "mlp_acc_pure",
                "bayes_acc_all",
                "weight_one",
            )
        }
        q = block.get("qelm")
        if q:
            out["case3"][n] = {
                k: (np.asarray(q[k]["mean"], float), np.asarray(q[k]["err"], float))
                for k in (
                    "M+state/ridge_acc",
                    "state/ridge_acc",
                    "M/ridge_acc",
                    "bayes_M_acc",
                    "M+state/ridge_acc_mixed",
                )
            }
    return out


def load_panels(path: Path) -> list[dict]:
    """``{title, curves: {N: (x, y, err)}}`` for each panel of a figure json."""
    d = json.loads(path.read_text())
    panels = []
    for pn in d["panels"]:
        curves = {}
        for c in pn["curves"]:
            if not c["label"].startswith("N = "):
                continue
            n = int(c["label"].split("=")[1])
            curves[n] = (
                np.asarray(c["x"], float),
                np.asarray(c["y"], float),
                np.asarray(c.get("yerr_upper", np.zeros(len(c["y"]))), float),
            )
        panels.append({"title": pn["title"], "curves": curves})
    return panels


def skill(acc, acc_err, bayes, bayes_err):
    """``g = (acc - 1/2) / (bayes - 1/2)``, nan where Bayes is within 0.01 of chance."""
    den = bayes - 0.5
    ok = den > 0.01
    g = np.full_like(acc, np.nan)
    ge = np.full_like(acc, np.nan)
    g[ok] = (acc[ok] - 0.5) / den[ok]
    ge[ok] = np.sqrt(
        (acc_err[ok] / den[ok]) ** 2 + (g[ok] * bayes_err[ok] / den[ok]) ** 2
    )
    return g, ge


# ---------------------------------------------------------------- figures


def _style(ax, symlog=True):
    if symlog:
        ax.set_xscale("symlog", linthresh=0.02, linscale=0.4)
        ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.set_xlabel("$p$")


def _pc_band(ax, lo=0.10, hi=0.16):
    ax.axvspan(lo, hi, color="0.85", zorder=0)


def fig1(dh: dict, out: Path, summary: dict) -> None:
    p = dh["p"]
    c1, c3 = dh["case1"], dh["case3"]
    n1, n3 = sorted(c1), sorted(c3)
    col1 = dict(zip(n1, colors(len(n1)), strict=True))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    ax = axes[0]
    for n in n1:
        y, e = c1[n]["bayes_acc_all"]
        ax.errorbar(
            p, y, e, color=col1[n], marker="s", ms=3, capsize=2, label=f"N = {n}"
        )
    ax.set_title("(a) Bayes sur $M$ (connaît le circuit)\n$(1+f)/2$")
    ax.set_ylabel("précision le long de l'axe de purification")

    ax = axes[1]
    for n in n1:
        y, e = c1[n]["ridge_acc_all"]
        ax.errorbar(
            p, y, e, color=col1[n], marker="o", ms=3, capsize=2, label=f"N = {n}"
        )
    ax.set_title("(b) QELM-M : ridge sur vec($M$)")

    ax = axes[2]
    for n in n3:
        y, e = c3[n]["M+state/ridge_acc"]
        ax.errorbar(
            p, y, e, color=col1[n], marker="o", ms=3, capsize=2, label=f"N = {n}, QELM"
        )
        yb, eb = c3[n]["bayes_M_acc"]
        ax.plot(p, yb, color=col1[n], ls=":", lw=1.2)
    ax.plot([], [], color="0.3", ls=":", label="Bayes sur $M$ (max 2/3)")
    ax.set_title(
        "(c) QELM complet : ridge sur\n$M$ + $\\langle P_k\\rangle$ + $\\langle P_kP_l\\rangle$"  # noqa: E501
    )
    ax.set_ylabel("précision moyenne sur $X, Y, Z$")

    ax = axes[3]
    for n in n1:
        acc, acc_e = c1[n]["ridge_acc_all"]
        b, be = c1[n]["bayes_acc_all"]
        g, ge = skill(acc, acc_e, b, be)
        ax.errorbar(
            p,
            g,
            ge,
            color=col1[n],
            marker="o",
            ms=3,
            capsize=2,
            label=f"N = {n}, ridge $M$",
        )
    for n in n3:
        acc, acc_e = c3[n]["M+state/ridge_acc"]
        b, be = c3[n]["bayes_M_acc"]
        g, ge = skill(acc, acc_e, b, be)
        m = p >= 0.08
        ax.errorbar(
            p[m],
            g[m],
            ge[m],
            color=col1[n],
            marker="^",
            ms=3,
            ls="--",
            capsize=2,
            label=f"N = {n}, QELM complet",
        )
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_title("(d) part de l'avantage de Bayes\n$g = (acc-1/2)/(acc_{Bayes}-1/2)$")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    for ax in axes:
        _style(ax)
        _pc_band(ax)
        ax.axhline(0.5, color="0.5", ls="--", lw=0.8) if ax is not axes[3] else None
    for ax in axes[:3]:
        ax.set_ylim(0.48, 1.02)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig1_meme_circuit.png", dpi=150)
    plt.close(fig)

    # crossings for the summary
    s = {}
    for name, key in (
        ("bayes_M", "bayes_acc_all"),
        ("purified_fraction", "purified"),
        ("ridge_M_raw", "ridge_acc_all"),
        ("mlp_M_raw", "mlp_acc_all"),
    ):
        s[name] = consecutive_crossings(p, {n: c1[n][key] for n in n1})
    g_curves = {}
    for n in n1:
        acc, acc_e = c1[n]["ridge_acc_pure"]
        g_curves[n] = (2 * acc - 1, 2 * acc_e)
    s["ridge_M_skill"] = consecutive_crossings(p, g_curves)
    s["qelm_full_raw"] = consecutive_crossings(
        p, {n: c3[n]["M+state/ridge_acc"] for n in n3}
    )
    gq = {}
    for n in n3:
        acc, acc_e = c3[n]["M+state/ridge_acc"]
        b, be = c3[n]["bayes_M_acc"]
        g, ge = skill(acc, acc_e, b, be)
        gq[n] = (np.nan_to_num(g, nan=1.0), np.nan_to_num(ge))
    s["qelm_full_skill"] = consecutive_crossings(p, gq, lo=0.05)
    s["qelm_state_only_raw"] = consecutive_crossings(
        p, {n: c3[n]["state/ridge_acc"] for n in n3}
    )
    # skill values near p_c, for the text
    i16 = int(np.argmin(abs(p - 0.16)))
    s["skill_at_p016"] = {
        "ridge_M": {n: float(g_curves[n][0][i16]) for n in n1},
        "qelm_full": {n: float(gq[n][0][i16]) for n in n3},
    }
    summary["clifford_dehghani"] = s


def fig2(out: Path, summary: dict) -> None:
    panels = load_panels(CHARGE)
    pick = [0, 2, 3]
    titles = [
        "(a) oracle : $4\\langle \\mathrm{Var}\\,Q\\rangle$ de l'état exact",
        "(b) QELM linéaire sur $\\langle Z_i\\rangle, \\langle Z_iZ_j\\rangle$\n(moyenne prédite ×4, poolé)",  # noqa: E501
        "(c) lecture quadratique de $\\langle Z_i\\rangle$\n(moyenne prédite ×4, poolé)",  # noqa: E501
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    s = {}
    for ax, k, t in zip(axes, pick, titles, strict=True):
        curves = panels[k]["curves"]
        ns = sorted(curves)
        cl = dict(zip(ns, colors(len(ns)), strict=True))
        for n in ns:
            x, y, e = curves[n]
            ax.errorbar(
                x, y, e, color=cl[n], marker="o", ms=3, capsize=2, label=f"N = {n}"
            )
        ax.set_title(t, fontsize=10)
        _style(ax, symlog=False)
        ax.axvspan(0.06, 0.11, color="0.88", zorder=0)
        x0 = curves[ns[0]][0]
        s[panels[k]["title"]] = consecutive_crossings(
            x0, {n: (curves[n][1], curves[n][2]) for n in ns}, lo=0.01, hi=0.4
        )
    axes[0].set_ylabel("paramètre d'ordre / 4 × prédiction moyenne")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_controles_u1.png", dpi=150)
    plt.close(fig)
    summary["u1_charge_variance"] = s


def fig3(out: Path, summary: dict) -> None:
    panels = load_panels(CLIFF_SCR)
    oracle, ridge = panels[0]["curves"], panels[1]["curves"]
    ns = sorted(oracle)
    cl = dict(zip(ns, colors(len(ns)), strict=True))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    s = {"shrinkage": {}}
    for n in ns:
        x, y, e = oracle[n]
        _, yh, eh = ridge[n]
        axes[0].errorbar(
            x, y, e, color=cl[n], marker="s", ms=3, capsize=2, label=f"N = {n}"
        )
        axes[1].errorbar(
            x, yh, eh, color=cl[n], marker="o", ms=3, capsize=2, label=f"N = {n}"
        )
        ybar = float(np.mean(y))  # equal number of trajectories per p
        lam, icpt = np.polyfit(y - ybar, yh - ybar, 1)
        resid = (yh - ybar) - (lam * (y - ybar) + icpt)
        axes[2].plot(
            y - ybar,
            yh - ybar,
            "o",
            color=cl[n],
            ms=4,
            label=f"N = {n} : pente λ = {lam:.2f}",
        )
        s["shrinkage"][n] = {
            "ybar": ybar,
            "lambda": float(lam),
            "intercept": float(icpt),
            "rms_residual": float(np.sqrt(np.mean(resid**2))),
            "oracle_minus_ybar_at_p016": float(np.interp(0.16, x, y) - ybar),
        }
    lim = 0.6
    axes[2].plot(
        [-lim, lim],
        [-lim, lim],
        color="k",
        lw=0.8,
        ls="--",
        label="pente 1 (lecture parfaite)",
    )
    axes[2].axhline(0, color="0.5", lw=0.6)
    axes[2].axvline(0, color="0.5", lw=0.6)
    axes[2].set_xlabel("$\\langle S_R\\rangle(p) - \\bar S$ (oracle)")
    axes[2].set_ylabel("$\\hat S(p) - \\bar S$ (ridge poolé)")
    axes[2].set_title("(c) la prédiction est l'oracle rétréci vers sa moyenne")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.25)
    for ax, t in zip(
        axes[:2],
        (
            "(a) oracle $\\langle S_R\\rangle$ (bits), Clifford brouillé",
            "(b) ridge poolé sur $|\\langle P\\rangle|$ : moyenne prédite",
        ),
        strict=True,
    ):
        _style(ax, symlog=False)
        ax.axvline(P_C_CLIFFORD, color="k", ls=":", lw=1)
        ax.set_title(t, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("$\\langle S_R\\rangle$ (bits)")
    fig.tight_layout()
    fig.savefig(out / "fig3_retrecissement.png", dpi=150)
    plt.close(fig)
    x = oracle[ns[0]][0]
    s["oracle_crossings"] = consecutive_crossings(
        x, {n: oracle[n][1:] for n in ns}, lo=0.02, hi=0.4
    )
    s["ridge_pooled_crossings"] = consecutive_crossings(
        x, {n: ridge[n][1:] for n in ns}, lo=0.02, hi=0.4
    )
    t2n = load_panels(CLIFF_T2N)[0]["curves"]
    nt = sorted(t2n)
    s["oracle_T2N_crossings"] = consecutive_crossings(
        t2n[nt[0]][0], {n: t2n[n][1:] for n in nt}, lo=0.02, hi=0.4
    )
    summary["clifford_state_readout"] = s


def fig4(out: Path, summary: dict) -> None:
    rows_a = [
        (
            "$\\langle S_R\\rangle$ oracle, brouillé (#292)",
            summary["clifford_state_readout"]["oracle_crossings"],
            "k",
            "s",
        ),
        (
            "$\\langle S_R\\rangle$ oracle, $T=2N$ (#292)",
            summary["clifford_state_readout"]["oracle_T2N_crossings"],
            "0.45",
            "s",
        ),
        (
            "Bayes sur $M$, $(1+f)/2$ (#287)",
            summary["clifford_dehghani"]["bayes_M"],
            "tab:blue",
            "D",
        ),
        (
            "ridge sur $M$, part $g$ (#287)",
            summary["clifford_dehghani"]["ridge_M_skill"],
            "tab:orange",
            "o",
        ),
        (
            "QELM complet, brut (#287)",
            summary["clifford_dehghani"]["qelm_full_raw"],
            "tab:green",
            "^",
        ),
        (
            "QELM complet, part $g$ (#287)",
            summary["clifford_dehghani"]["qelm_full_skill"],
            "tab:olive",
            "v",
        ),
        (
            "ridge poolé $|\\langle P\\rangle|$, brouillé (#292)",
            summary["clifford_state_readout"]["ridge_pooled_crossings"],
            "tab:red",
            "x",
        ),
    ]
    u1 = summary["u1_charge_variance"]
    keys = list(u1)
    rows_b = [
        ("$4\\langle\\mathrm{Var}\\,Q\\rangle$ oracle (#299)", u1[keys[0]], "k", "s"),
        ("QELM linéaire poolé (#299)", u1[keys[1]], "tab:orange", "o"),
        ("lecture quadratique poolée (#299)", u1[keys[2]], "tab:green", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, rows, ref, reflab in (
        (axes[0], rows_a, P_C_CLIFFORD, "$p_c = 0.16$ (Clifford, Gullans & Huse)"),
        (
            axes[1],
            rows_b,
            P_SHARP,
            "$p_\\# \\approx 0.09$ (ordre de grandeur, Barratt et al.)",
        ),
    ):
        for j, (lab, cr, col, mk) in enumerate(rows):
            off = (j - len(rows) / 2) * 0.0015
            xs, ys, es, filled, none = [], [], [], [], []
            for pair, r in cr.items():
                a, b = (int(v) for v in pair.split("-"))
                inv = 1.0 / np.sqrt(a * b) + off
                if r["resolved"] is not None:
                    xs.append(inv)
                    ys.append(r["resolved"]["p_star"])
                    es.append(r["p_star_err"] or 0.0)
                    filled.append(True)
                elif r["mean_curve_crossings"]:
                    xs.append(inv)
                    ys.append(
                        r["p_star"]
                        if r["p_star"] is not None
                        else r["mean_curve_crossings"][0]
                    )
                    es.append(r["p_star_err"] or 0.0)
                    filled.append(False)
                else:
                    none.append(inv)
            ax.plot(xs, ys, color=col, lw=0.8, label=lab, marker=mk, ms=0)
            for x, y, e, fl in zip(xs, ys, es, filled, strict=True):
                ax.errorbar(
                    [x],
                    [y],
                    [e],
                    color=col,
                    marker=mk,
                    ms=7,
                    capsize=2,
                    mfc=col if fl else "white",
                )
            if none:
                ax.plot(
                    none,
                    [0.008 + 0.01 * j] * len(none),
                    marker="|",
                    color=col,
                    ms=12,
                    ls="none",
                )
        ax.axhline(ref, color="k", ls=":", lw=1, label=reflab)
        ax.set_xlabel("$1/\\sqrt{N N'}$  (paire de tailles consécutives)")
        ax.set_ylabel("$p^*$ du croisement")
        ax.set_ylim(0, 0.42)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_title("(a) purification de $R$, circuits de Clifford")
    axes[1].set_title("(b) charge sharpening, circuits U(1)")
    for ax in axes:
        ax.text(
            0.01,
            0.98,
            "plein : croisement résolu (écart > 2σ des deux côtés) ; creux : non résolu ;\nbarres en bas : aucun croisement des courbes moyennes",  # noqa: E501
            transform=ax.transAxes,
            fontsize=7,
            color="0.3",
            va="top",
        )
    fig.tight_layout()
    fig.savefig(out / "fig4_critere_fss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    dh = load_dehghani()
    fig1(dh, args.out, summary)
    fig2(args.out, summary)
    fig3(args.out, summary)
    fig4(args.out, summary)
    (args.out / "verdict_summary.json").write_text(
        json.dumps(summary, indent=1, default=str)
    )
    print(json.dumps(summary, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
