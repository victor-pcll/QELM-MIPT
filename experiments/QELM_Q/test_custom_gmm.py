#!/usr/bin/env python3
"""Parity of the NumPy Gaussian mixture with scikit-learn on a sweep (issue #223).

The Gaussian-mixture ceiling of the Bayes accuracy and its BIC choice of the
number ``K`` of Gaussians per charge sector (#208, #215) were fitted with
``sklearn.mixture.GaussianMixture``. Before :class:`QELM.evaluation.gmm.
CustomGMM` replaces it, this script refits both on the runs of an
``exp_QELM_Q.sh`` sweep (the ``N = 8`` plane of #215) and compares them.

For every run, on the split of the run (the first ``train_size`` samples of
``exp_QELM_Q.npz`` to train, the rest to test, the features ``<Z_k>``
without the bias column) and with the seed of the run, the training samples
of each charge sector are fitted with ``K = 1 .. 8`` Gaussians three ways:

- ``sklearn``: ``GaussianMixture(K, covariance_type="full",
  random_state=seed)``, the model of the logged ``bayes_estimate_gmm``
  (the script checks that it finds the logged number at ``K = 3``);
- ``shared``: :class:`CustomGMM` started from the same k-means partition as
  scikit-learn (``KMeans(K, n_init=1, random_state=seed)``). Only the EM
  rounds differ in code, so this tests the E-step, the M-step, the
  regularization, the stopping rule and the BIC: the log-likelihoods and BIC
  must agree to rounding, the selected ``K`` and the predicted labels
  exactly;
- ``numpy``: :class:`CustomGMM` with its own k-means++ initialization and the
  same seed, the fully independent fit. Its k-means seeds are not those of
  scikit-learn, so EM can stop at another local maximum of the likelihood:
  this measures how much the ceilings depend on the start.

For each way, the run's numbers are: the total training log-likelihood
``ln L`` and the BIC ``nu ln n - 2 ln L`` of every (sector, ``K``), the ``K``
of smallest BIC in each sector, and the test accuracy of the Bayes rule
``argmax_c [log f_c(x) + log pi_c]`` with ``K = 3`` in both sectors
(``accuracy_K3``) and with the BIC ``K`` of each sector (``accuracy_bic``).

Files, in ``--out`` (default: the sweep folder), each figure with its JSON
(:func:`QELM.utils.plots.save_figure`):

- ``gmm_parity_scatter.png``: for every run, sector and ``K``, the gap of
  ``ln L / n`` and of the BIC to scikit-learn, of ``shared`` (rounding,
  ``~1e-12``) and of ``numpy`` (local maxima);
- ``gmm_parity_heatmaps.png``: over ``(p, h)``, averaged over the seeds, the
  gaps ``Delta accuracy`` (``K = 3`` and BIC) to scikit-learn and the
  fraction of sectors where both select the same ``K``, for ``shared`` and
  ``numpy``;
- ``gmm_parity_runs.json``: every number of every run, and
  ``gmm_parity_summary.json``: the averages over the seeds and the largest
  gaps of the whole sweep.

The selected cuts of the issue (``h = 1 J``, ``p = 1`` and ``p = 0.5``) are
printed as a table at the end. The sweep of #215, then this script:

    N_VALUES=8 H_VALUES="0.5 1 2 5 10" P_VALUES="0 0.1 0.2 0.3 0.5 0.7 1" \\
        SEEDS="42 43 44" L=500 STATE=vector ./experiments/QELM_Q/exp_QELM_Q.sh
    uv run python experiments/QELM_Q/test_custom_gmm.py \\
        outputs/exp_QELM_Q/<date_time>

The file name is the one of the issue; it is a script, not a pytest module
(``pytest`` only collects ``tests/``, where ``tests/test_custom_gmm.py``
checks the same parity on synthetic data).
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from QELM.evaluation.gmm import CustomGMM
from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    color_map,
    draw_heatmap,
    save_figure,
    subplots,
    sweep_grid,
)

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("h", "TFIM.h"),
)
NPZ = "exp_QELM_Q.npz"
K_MAX = 8
K_FIXED = 3
# The custom fits compared with scikit-learn.
WAYS = ("shared", "numpy")
METRICS = tuple(
    f"{name}_{way}"
    for way in WAYS
    for name in (
        "delta_accuracy_K3",
        "delta_accuracy_bic",
        "same_k_bic",
        "same_labels_bic",
        "max_abs_delta_loglik",
        "max_abs_delta_bic",
    )
) + ("accuracy_K3_sklearn", "accuracy_bic_sklearn")
# (metric, title, color map, (vmin, vmax)) of the maps.
MAPS = (
    ("delta_accuracy_K3_shared", "shared start: Δ accuracy, K = 3", "RdBu_r", 0.05),
    ("delta_accuracy_bic_shared", "shared start: Δ accuracy, BIC", "RdBu_r", 0.05),
    ("same_k_bic_shared", "shared start: same BIC K", "viridis", None),
    ("delta_accuracy_K3_numpy", "own k-means: Δ accuracy, K = 3", "RdBu_r", 0.05),
    ("delta_accuracy_bic_numpy", "own k-means: Δ accuracy, BIC", "RdBu_r", 0.05),
    ("same_k_bic_numpy", "own k-means: same BIC K", "viridis", None),
)
# The cuts printed at the end: (h, p).
CUTS = ((1.0, 0.5), (1.0, 1.0))


def kmeans_resp(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """The one-hot k-means partition that ``GaussianMixture`` starts from."""
    labels = KMeans(n_clusters=K, n_init=1, random_state=seed).fit(X).labels_
    resp = np.zeros((len(X), K))
    resp[np.arange(len(X)), labels] = 1.0
    return resp


def fit_sector(X: np.ndarray, seed: int, k_max: int = K_MAX) -> dict[str, list[Any]]:
    """Fit ``K = 1 .. k_max`` Gaussians to one sector, the three ways.

    Returns:
        For each way (``sklearn``, ``shared``, ``numpy``), the fitted mixtures
        of ``K = 1 .. min(k_max, n)``.
    """
    fits: dict[str, list[Any]] = {"sklearn": [], "shared": [], "numpy": []}
    for K in range(1, min(k_max, len(X)) + 1):
        fits["sklearn"].append(
            GaussianMixture(
                n_components=K, covariance_type="full", random_state=seed
            ).fit(X)
        )
        fits["shared"].append(CustomGMM(K).fit(X, resp_init=kmeans_resp(X, K, seed)))
        fits["numpy"].append(CustomGMM(K, random_state=seed).fit(X))
    return fits


def bayes_predict(
    mixtures: Sequence[Any], priors: np.ndarray, classes: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """``argmax_c [log f_c(x) + log pi_c]``, one fitted mixture per class."""
    delta = np.stack(
        [
            m.score_samples(X) + np.log(pi)
            for m, pi in zip(mixtures, priors, strict=True)
        ],
        axis=1,
    )
    return np.asarray(classes[np.argmax(delta, axis=1)])


def compare_run(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Refit the three ways on one run and compare them with scikit-learn.

    Returns:
        The metrics of :data:`METRICS`, plus ``points``: one entry per
        (sector, ``K``) with the gaps of ``ln L / n`` and of the BIC.

    Raises:
        RuntimeError: If scikit-learn at ``K = 3`` does not give the logged
            ``bayes_estimate_gmm`` of the run.
    """
    seed = int(row["seed"])
    with np.load(run_dir / NPZ) as data:
        X, Y, n_test = data["X"][:, :-1], data["Y"], len(data["y_test"])
    train = len(Y) - n_test
    X_tr, y_tr, X_te, y_te = X[:train], Y[:train], X[train:], Y[train:]
    classes, counts = np.unique(y_tr, return_counts=True)
    priors = counts / train
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings, counted below
        for c in classes:
            fits[c] = fit_sector(X_tr[y_tr == c], seed)

    out: dict[str, Any] = {"points": []}
    k_bic: dict[str, list[int]] = {way: [] for way in ("sklearn", *WAYS)}
    for c in classes:
        X_c = X_tr[y_tr == c]
        n_c = len(X_c)
        bic = {way: np.array([m.bic(X_c) for m in fits[c][way]]) for way in k_bic}
        for way in k_bic:
            k_bic[way].append(int(np.argmin(bic[way])))
        for K, sk in enumerate(fits[c]["sklearn"], start=1):
            point: dict[str, Any] = {"sector": float(c), "K": K}
            for way in WAYS:
                ours = fits[c][way][K - 1]
                point[f"delta_loglik_{way}"] = ours.score(X_c) - sk.score(X_c)
                point[f"delta_bic_{way}"] = float(
                    bic[way][K - 1] - bic["sklearn"][K - 1]
                )
                point[f"converged_{way}"] = bool(ours.converged_)
            point["n"] = n_c
            out["points"].append(point)

    def predict(way: str, ks: Sequence[int]) -> np.ndarray:
        mixtures = [fits[c][way][k] for c, k in zip(classes, ks, strict=True)]
        return bayes_predict(mixtures, priors, classes, X_te)

    can_k3 = counts.min() >= K_FIXED
    fixed = [K_FIXED - 1] * len(classes)
    pred = {
        way: {
            "K3": predict(way, fixed) if can_k3 else None,
            "bic": predict(way, k_bic[way]),
        }
        for way in k_bic
    }
    acc = {
        way: {
            rule: float(np.mean(p == y_te)) if p is not None else float("nan")
            for rule, p in rules.items()
        }
        for way, rules in pred.items()
    }
    logged = float(row["bayes_estimate_gmm"])
    if not (
        np.isclose(acc["sklearn"]["K3"], logged, rtol=0, atol=1e-12)
        or (np.isnan(logged) and np.isnan(acc["sklearn"]["K3"]))
    ):
        raise RuntimeError(
            f"{run_dir.name}: refitted scikit-learn K = 3 gives "
            f"{acc['sklearn']['K3']}, the run logged {logged}."
        )
    out["accuracy_K3_sklearn"] = acc["sklearn"]["K3"]
    out["accuracy_bic_sklearn"] = acc["sklearn"]["bic"]
    for way in WAYS:
        out[f"accuracy_K3_{way}"] = acc[way]["K3"]
        out[f"accuracy_bic_{way}"] = acc[way]["bic"]
        out[f"delta_accuracy_K3_{way}"] = acc[way]["K3"] - acc["sklearn"]["K3"]
        out[f"delta_accuracy_bic_{way}"] = acc[way]["bic"] - acc["sklearn"]["bic"]
        out[f"k_bic_{way}"] = [k + 1 for k in k_bic[way]]
        out[f"same_k_bic_{way}"] = float(
            np.mean(np.array(k_bic[way]) == np.array(k_bic["sklearn"]))
        )
        out[f"same_labels_bic_{way}"] = float(
            np.mean(pred[way]["bic"] == pred["sklearn"]["bic"])
        )
        out[f"max_abs_delta_loglik_{way}"] = max(
            abs(p[f"delta_loglik_{way}"]) for p in out["points"]
        )
        out[f"max_abs_delta_bic_{way}"] = max(
            abs(p[f"delta_bic_{way}"]) for p in out["points"]
        )
    out["k_bic_sklearn"] = [k + 1 for k in k_bic["sklearn"]]
    return out


def draw_scatter(rows: Sequence[dict[str, Any]]) -> Figure:
    """Gaps of ``ln L / n`` and of the BIC to scikit-learn, against ``K``."""
    colors = color_map(WAYS)
    fig, axes = subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    rng = np.random.default_rng(0)
    for row in rows:
        for point in row["points"]:
            jitter = point["K"] + 0.15 * rng.uniform(-1, 1)
            axes[0, 0].plot(
                jitter,
                max(abs(point["delta_loglik_shared"]), 1e-17),
                ".",
                color=colors["shared"],
                alpha=0.4,
            )
            axes[0, 1].plot(
                jitter,
                max(abs(point["delta_bic_shared"]), 1e-17),
                ".",
                color=colors["shared"],
                alpha=0.4,
            )
            axes[0, 2].plot(
                jitter,
                point["delta_loglik_numpy"],
                ".",
                color=colors["numpy"],
                alpha=0.4,
            )
    axes[0, 0].set(
        yscale="log",
        xlabel=r"Gaussians per sector $K$",
        ylabel=r"$|\Delta \ln L| / n$",
        title="shared start: log-likelihood",
    )
    axes[0, 1].set(
        yscale="log",
        xlabel=r"Gaussians per sector $K$",
        ylabel=r"$|\Delta \mathrm{BIC}|$",
        title="shared start: BIC",
    )
    axes[0, 2].axhline(0.0, color="gray", lw=1)
    axes[0, 2].set(
        yscale="symlog",
        xlabel=r"Gaussians per sector $K$",
        ylabel=r"$(\ln L_\mathrm{numpy} - \ln L_\mathrm{sklearn}) / n$",
        title="own k-means: log-likelihood",
    )
    axes[0, 2].set_yscale("symlog", linthresh=1e-6)
    fig.suptitle(
        "CustomGMM minus scikit-learn, every run, sector and K (training samples)"
    )
    return fig


def draw_maps(summary: Sequence[dict[str, Any]], N: int) -> Figure:
    """The maps of :data:`MAPS` over ``(p, h)``, averaged over the seeds."""
    hs = [float(r["h"]) for r in summary]
    log_h = min(hs) > 0 and max(hs) / min(hs) > 4
    fig, axes = subplots(2, 3, figsize=(15.6, 8.4), squeeze=False)
    for ax, (metric, title, cmap, span) in zip(axes.flat, MAPS, strict=True):
        grid = sweep_grid(summary, "p", "h", metric)
        vmin, vmax = (-span, span) if span is not None else (0.0, 1.0)
        draw_heatmap(
            ax, grid, label=metric, log_y=log_h, vmin=vmin, vmax=vmax, cmap=cmap
        )
        for j, h in enumerate(grid.y):
            for i, p in enumerate(grid.x):
                if np.isfinite(grid.mean[j, i]):
                    ax.text(
                        p,
                        h,
                        f"{grid.mean[j, i]:+.2f}"[span is None :],
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
        ax.set(xlabel=r"$p$", ylabel=r"$h / J$", title=title)
    fig.suptitle(f"CustomGMM minus scikit-learn, test accuracy on $Q$, N = {N}")
    return fig


def analyze_sweep(sweep_dir: Path, out_dir: Path | None = None) -> dict[str, Path]:
    """Compare the three ways on every run of a sweep; write figures and tables."""
    out_dir = out_dir or sweep_dir
    rows = load_sweep(sweep_dir, keys=KEYS)
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['run']}", flush=True)
        row.update(compare_run(sweep_dir / row["run"], row))
    Ns = sorted({int(r["N"]) for r in rows})
    if len(Ns) != 1:
        raise ValueError(f"The sweep has several N = {Ns}: one at a time.")
    summary = summarize(rows, by=("N", "p", "h"), metrics=METRICS)
    totals = {
        (key if key.startswith("max_abs") else f"max_abs_{key}"): max(
            abs(float(r[key])) for r in rows if np.isfinite(r[key])
        )
        for key in METRICS
        if key.startswith(("delta", "max_abs"))
    }
    for way in WAYS:
        points = [p for r in rows for p in r["points"]]
        totals[f"fraction_same_k_bic_{way}"] = float(
            np.mean([r[f"same_k_bic_{way}"] for r in rows])
        )
        totals[f"fraction_same_labels_bic_{way}"] = float(
            np.mean([r[f"same_labels_bic_{way}"] for r in rows])
        )
        totals[f"fraction_converged_{way}"] = float(
            np.mean([p[f"converged_{way}"] for p in points])
        )
        totals[f"fraction_lower_loglik_{way}"] = float(
            np.mean([p[f"delta_loglik_{way}"] < -1e-9 for p in points])
        )
        totals[f"fraction_higher_loglik_{way}"] = float(
            np.mean([p[f"delta_loglik_{way}"] > 1e-9 for p in points])
        )
    out: dict[str, Path] = {}
    meta = {"N": Ns[0], "k_max": K_MAX, "k_fixed": K_FIXED, "n_runs": len(rows)}
    for key, (name, fig) in {
        "scatter": ("gmm_parity_scatter", draw_scatter(rows)),
        "heatmaps": ("gmm_parity_heatmaps", draw_maps(summary, Ns[0])),
    }.items():
        out[key] = save_figure(fig, out_dir, name, data=meta)
        plt.close(fig)
    out["runs"] = out_dir / "gmm_parity_runs.json"
    out["runs"].write_text(json.dumps(rows, indent=1, default=float) + "\n")
    out["summary"] = out_dir / "gmm_parity_summary.json"
    out["summary"].write_text(
        json.dumps({**meta, "totals": totals, "summary": summary}, indent=2) + "\n"
    )
    print_cuts(summary)
    print(json.dumps(totals, indent=2))
    return out


def print_cuts(summary: Sequence[dict[str, Any]]) -> None:
    """Print the accuracies at the cuts :data:`CUTS`, if they are in the sweep."""
    header = (
        f"{'h':>4} {'p':>4} | {'sk K3':>6} {'sh K3':>6} {'np K3':>6} | "
        f"{'sk BIC':>6} {'sh BIC':>6} {'np BIC':>6} | same K sh/np"
    )
    print(header)
    for h, p in CUTS:
        for r in summary:
            if np.isclose(float(r["h"]), h) and np.isclose(float(r["p"]), p):
                sk3 = r["accuracy_K3_sklearn_mean"]
                skb = r["accuracy_bic_sklearn_mean"]
                print(
                    f"{h:>4g} {p:>4g} | {sk3:6.3f} "
                    f"{sk3 + r['delta_accuracy_K3_shared_mean']:6.3f} "
                    f"{sk3 + r['delta_accuracy_K3_numpy_mean']:6.3f} | {skb:6.3f} "
                    f"{skb + r['delta_accuracy_bic_shared_mean']:6.3f} "
                    f"{skb + r['delta_accuracy_bic_numpy_mean']:6.3f} | "
                    f"{r['same_k_bic_shared_mean']:.2f}/{r['same_k_bic_numpy_mean']:.2f}"
                )


def main() -> None:
    """Parse the command line and compare one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/exp_QELM_Q",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output folder; default: the sweep"
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "exp_QELM_Q")
    for path in analyze_sweep(sweep_dir, args.out).values():
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
