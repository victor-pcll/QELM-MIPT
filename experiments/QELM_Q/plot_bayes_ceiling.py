#!/usr/bin/env python3
"""Draw the Bayes ceilings of a p-sweep of exp_QELM_Q (issue #202).

Reads the run folders written by ``exp_QELM_Q.sh`` (one per ``N``, seed and
``p``), averages over the seeds and draws five figures in the sweep folder:

- ``bayes_ceiling.png``: the accuracy of the ridge readout of ``Q`` from
  ``<Z_k>`` against ``p``, one solid curve per size ``N`` (finite-size
  scaling), with, in the same color, its three ceilings: the QDA
  (``bayes_estimate_qda``, dashed), the mixture of Gaussians
  (``bayes_estimate_gmm``, dotted) and the K-NN (``bayes_estimate_knn``,
  dash-dotted), and chance (0.5, gray dotted). ``--no-ceilings`` leaves the
  ceilings out.
- Four figures on the PCA plane, one panel per ``(N, p)``: the test
  ``<Z_k>`` of one run projected on their plane of largest variance,
  colored by charge sector, with

  - ``qda_contours.png``: the 1 and 2 sigma ellipses of the Gaussian
    ``f_k`` fitted by the QDA to each sector;
  - ``gmm_contours.png``: the contour lines of the mixture of Gaussians
    fitted to each sector, and the 1 sigma ellipse of each component;
  - ``ridge_regions.png`` and ``knn_regions.png``: the decision regions of
    the ridge readout and of the K-NN on the plane, with the test samples
    they get wrong crossed.

Reading them together: where the ridge accuracy falls to 0.5,

- ceilings at 0.5 and densities on top of each other: the sectors overlap,
  the information is erased (decoherence);
- ceilings well above 0.5 and densities apart: the information is in
  ``<Z_k>`` but not linearly (learnability transition): the ridge boundary
  is a straight line through clusters that the K-NN regions and the mixture
  contours follow.

Every model is refitted from the ``.npz`` of the run, on the split of the
run (and with its seed for the mixture), so it is the model behind the
ceiling it illustrates; the ridge readout is the weights ``W`` of the run.
Each figure is written with its data (:func:`QELM.utils.plots.save_figure`).

The sweep must hold one value of ``TFIM.h``, ``evolution.t`` and
``final_measurement``, or the one to plot is chosen with ``--h``, ``--t``
and ``--final``. 500 samples per run (``L=500``) give clusters dense enough
to read at ``p = 1``:

    N_VALUES="4 6" H_VALUES=1 P_VALUES="0 0.5 1" L=500 \\
        ./experiments/QELM_Q/exp_QELM_Q.sh
    uv run python experiments/QELM_Q/plot_bayes_ceiling.py \\
        outputs/exp_QELM_Q/<date_time> --p 0 0.5 1
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.evaluation.bayes_optimal import (
    GMM_COMPONENTS,
    QDA,
    QDA_REG_PARAM,
    GaussianMixtureBayes,
    knn_classifier,
)
from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    COLORS,
    color_map,
    draw_companion,
    draw_curve,
    draw_decision_regions,
    draw_gmm_contours,
    draw_qda_contours,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("h", "TFIM.h"),
    ("t", "evolution.t"),
    ("final", "projective_measurement.final_measurement"),
)
# Parameters that the plotted runs must share.
FIXED = ("h", "t", "final")
METRICS = (
    "accuracy",
    "bayes_estimate_qda",
    "bayes_estimate_gmm",
    "bayes_estimate_knn",
)
# (metric, legend name, line style) of the ceilings.
CEILINGS = (
    ("bayes_estimate_qda", "QDA", "--"),
    ("bayes_estimate_gmm", "GMM", ":"),
    ("bayes_estimate_knn", "K-NN", "-."),
)
# Figures on the PCA plane: name -> (file name, title).
PLANES = {
    "qda": ("qda_contours", "QDA densities"),
    "gmm": ("gmm_contours", f"Mixtures of {GMM_COMPONENTS} Gaussians"),
    "ridge": ("ridge_regions", "Ridge decision regions"),
    "knn": ("knn_regions", "K-NN decision regions"),
}
NPZ = "exp_QELM_Q.npz"


def select_runs(
    rows: Sequence[dict[str, Any]], **fixed: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the runs with the given ``h``, ``t``, ``final``; check they are unique.

    Args:
        rows: Output of :func:`QELM.utils.load_sweep` with :data:`KEYS`.
        fixed: Values of :data:`FIXED` to keep; ``None`` keeps every value.

    Returns:
        The kept rows and the single value of each key of :data:`FIXED`.

    Raises:
        ValueError: If no row is left, or a key of :data:`FIXED` still takes
            several values.
    """
    kept = [
        r
        for r in rows
        if all(v is None or r[k] == v for k, v in fixed.items() if k in FIXED)
    ]
    if not kept:
        raise ValueError(f"No run with {fixed}.")
    values = {}
    for key in FIXED:
        found = sorted({r[key] for r in kept}, key=str)
        if len(found) > 1:
            raise ValueError(f"The sweep has several {key} = {found}: pick one.")
        values[key] = found[0]
    return kept, values


def subtitle(values: dict[str, Any]) -> str:
    """Return the fixed parameters of a figure as one line."""
    final = ", last round after the last step" if values["final"] else ""
    return f"h = {values['h']:g} J, t = {values['t']:g} / J{final}"


def draw_ceilings(
    rows: Sequence[dict[str, Any]], values: dict[str, Any], ceilings: bool = True
) -> Figure:
    """Draw the ridge accuracy against ``p``, one curve per ``N``, and its ceilings.

    Args:
        rows: Runs of one ``(h, t, final)``, from :func:`select_runs`.
        values: Their ``h``, ``t`` and ``final``.
        ceilings: Draw the QDA, GMM and K-NN ceilings of each size, in its
            color.

    Returns:
        The open figure.
    """
    summary = summarize(rows, by=("N", "p"), metrics=METRICS)
    n_values = sizes(summary)
    colors = color_map(f"N{N}" for N in n_values)
    fig, ax = subplots(figsize=(7.5, 5.5))
    for N in n_values:
        color = colors[f"N{N}"]
        draw_curve(
            ax,
            sweep_curve(summary, N, "p", "accuracy"),
            color,
            label=f"N = {N}, ridge",
        )
        if not ceilings:
            continue
        for metric, name, style in CEILINGS:
            draw_companion(
                ax,
                summary,
                N,
                "p",
                metric,
                color,
                label=f"N = {N}, {name}",
                linestyle=style,
            )
    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="chance")
    ax.set(
        xlabel=r"measurement probability $p$",
        ylabel=r"accuracy on $Q$ from $\langle Z_k \rangle$",
        title=subtitle(values),
    )
    ax.legend(fontsize=9, ncol=2 if ceilings else 1)
    fig.suptitle("Ridge accuracy and Bayes ceilings")
    return fig


@dataclass(frozen=True)
class RunFit:
    """The models of one run, refitted on its split.

    Attributes:
        qda: The QDA of ``bayes_estimate_qda``.
        gmm: The mixture of ``bayes_estimate_gmm``.
        knn: The K-NN of ``bayes_estimate_knn``.
        W: The ridge weights of the run, bias last, shape ``(d + 1,)``.
        threshold: Midpoint of the two charges: the ridge readout predicts
            the larger charge at or above it.
        X_test: The test ``<Z_k>`` (bias column dropped), ``(n_test, d)``.
        y_test: Their charges, ``(n_test,)``.
    """

    qda: QDA
    gmm: GaussianMixtureBayes
    knn: Any
    W: np.ndarray
    threshold: float
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def classes(self) -> np.ndarray:
        """The two charges, sorted."""
        return np.asarray(self.qda.classes_)

    def predict_ridge(self, X: np.ndarray) -> np.ndarray:
        """Charge given by the ridge readout, as ``accuracy`` counts it."""
        y = np.hstack([X, np.ones((len(X), 1))]) @ self.W
        low, high = self.classes
        return np.where(y >= self.threshold, high, low)


def fit_run(run_dir: Path, seed: int) -> RunFit:
    """Refit the models of a run on its training samples.

    Args:
        run_dir: Folder of one run, holding :data:`NPZ`.
        seed: Seed of the run, the ``random_state`` of its mixture.

    Returns:
        The :class:`RunFit`.
    """
    with np.load(run_dir / NPZ) as data:
        X, Y, n_test = data["X"][:, :-1], data["Y"], len(data["y_test"])
        W = np.ravel(data["W"])
    train = len(Y) - n_test
    X_train, y_train = X[:train], Y[:train]
    qda = QDA(reg_param=QDA_REG_PARAM, tol=0.0).fit(X_train, y_train)
    gmm = GaussianMixtureBayes(n_components=GMM_COMPONENTS, random_state=seed)
    knn = knn_classifier(train).fit(X_train, y_train)
    return RunFit(
        qda=qda,
        gmm=gmm.fit(X_train, y_train),
        knn=knn,
        W=W,
        threshold=float(np.mean(qda.classes_)),
        X_test=X[train:],
        y_test=Y[train:],
    )


def draw_panel(ax: Any, fit: RunFit, method: str) -> dict[str, Any]:
    """Draw one model of a run on the PCA plane of its test samples.

    Args:
        ax: Axis to draw on.
        fit: The models of the run.
        method: A key of :data:`PLANES`.

    Returns:
        What the drawing function of :mod:`QELM.utils.plots.bayes` returns.
    """
    names = [f"Q = {q:g}" for q in fit.classes]
    data = (ax, fit.X_test, fit.y_test)
    if method == "qda":
        return draw_qda_contours(*data, fit.qda, names=names, colors=COLORS[:2])
    if method == "gmm":
        return draw_gmm_contours(*data, fit.gmm, names=names, colors=COLORS[:2])
    predict = {"ridge": fit.predict_ridge, "knn": fit.knn.predict}[method]
    return draw_decision_regions(
        *data, predict, fit.classes, names=names, colors=COLORS[:2]
    )


def draw_plane(
    sweep_dir: Path,
    rows: Sequence[dict[str, Any]],
    values: dict[str, Any],
    method: str = "qda",
    p_values: Sequence[float] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Draw one model on the PCA plane, one panel per ``(N, p)``.

    Args:
        sweep_dir: Folder of the sweep.
        rows: Runs of one ``(h, t, final)``, from :func:`select_runs`.
        values: Their ``h``, ``t`` and ``final``.
        method: A key of :data:`PLANES`.
        p_values: Values of ``p`` to draw, default every value of the
            sweep. A point with several seeds draws the smallest seed.

    Returns:
        The open figure and, per panel ``"N<N>_p<p>"``, the run drawn, its
        ridge accuracy and ceilings, and what :func:`draw_panel` returns.
    """
    n_values = sorted({int(r["N"]) for r in rows})
    ps = sorted({float(r["p"]) for r in rows} if p_values is None else p_values)
    fig, axes = subplots(
        len(n_values),
        len(ps),
        figsize=(4.2 * len(ps), 3.8 * len(n_values)),
        squeeze=False,
    )
    extra: dict[str, Any] = {}
    for i, N in enumerate(n_values):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            runs = sorted(
                (r for r in rows if int(r["N"]) == N and np.isclose(r["p"], p)),
                key=lambda r: r["seed"],
            )
            if not runs:
                ax.set_axis_off()
                continue
            run = runs[0]
            fit = fit_run(sweep_dir / run["run"], int(run["seed"]))
            drawn = draw_panel(ax, fit, method)
            scores = ", ".join(
                f"{name} {run[metric]:.2f}"
                for metric, name in (
                    ("accuracy", "ridge"),
                    *((m, n) for m, n, _ in CEILINGS),
                )
            )
            ax.set_title(f"N = {N}, p = {p:g}\n{scores}", fontsize=10)
            extra[f"N{N}_p{p:g}"] = {
                "run": run["run"],
                **{metric: run[metric] for metric in METRICS},
                **drawn,
            }
    axes[0, 0].legend(fontsize=8)
    title = PLANES[method][1]
    fig.suptitle(
        rf"{title} on the PCA plane of $\langle Z_k \rangle$, {subtitle(values)}"
    )
    return fig, extra


def plot_sweep(
    sweep_dir: Path,
    ceilings: bool = True,
    p_values: Sequence[float] | None = None,
    **fixed: Any,
) -> dict[str, Path]:
    """Write ``bayes_ceiling.png`` and the figures of :data:`PLANES`.

    Args:
        sweep_dir: Folder of the sweep.
        ceilings: As in :func:`draw_ceilings`.
        p_values: As in :func:`draw_plane`.
        fixed: ``h``, ``t``, ``final`` to keep, as in :func:`select_runs`.

    Returns:
        The path of each PNG file, ``"ceiling"`` and the keys of
        :data:`PLANES`; each has its ``.json`` next to it.
    """
    rows, values = select_runs(load_sweep(sweep_dir, keys=KEYS), **fixed)
    fig = draw_ceilings(rows, values, ceilings)
    out = {"ceiling": save_figure(fig, sweep_dir, "bayes_ceiling", data=values)}
    plt.close(fig)
    for method, (name, _) in PLANES.items():
        fig, extra = draw_plane(sweep_dir, rows, values, method, p_values)
        out[method] = save_figure(fig, sweep_dir, name, data=extra)
        plt.close(fig)
    return out


def main() -> None:
    """Parse the command line and plot one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/exp_QELM_Q",
    )
    parser.add_argument("--h", type=float, default=None, help="TFIM.h to plot")
    parser.add_argument("--t", type=float, default=None, help="evolution.t to plot")
    parser.add_argument(
        "--final",
        choices=("true", "false"),
        default=None,
        help="final_measurement to plot",
    )
    parser.add_argument(
        "--p", type=float, nargs="+", default=None, help="p of the PCA panels"
    )
    parser.add_argument(
        "--no-ceilings",
        action="store_true",
        help="draw the ridge accuracy only, without the ceilings",
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "exp_QELM_Q")
    final = None if args.final is None else args.final == "true"
    for png in plot_sweep(
        sweep_dir,
        ceilings=not args.no_ceilings,
        p_values=args.p,
        h=args.h,
        t=args.t,
        final=final,
    ).values():
        print(f"Figure: {png}")


if __name__ == "__main__":
    main()
