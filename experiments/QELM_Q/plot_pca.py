#!/usr/bin/env python3
"""PCA of the feature spaces [M], [M, Z] and [Z] of exp_QELM_Q (issue #225).

The ridge readout of the charge ``Q`` is linear: it can only tell the two
charge sectors apart along a direction of the feature space. This script
shows how the geometry of three feature spaces changes with the measurement
probability ``p``, for one size ``N`` (default 8):

- ``M``: the record of the intermediate measurements, ``(n_rounds * Ns)``
  entries, +-1 for a measured qubit and 0 for an unmeasured one
  (``features_M`` of ``exp_QELM_Q.npz``). At ``p = 0`` nothing is measured
  and the record is all zeros: the panel says so instead of drawing
  arbitrary axes (:mod:`QELM.evaluation.pca`).
- ``M_Z``: the record next to the final ``<Z_k>``, ``phi = [M, Z]``
  (``features_M_Z``).
- ``Z``: the final ``<Z_k>`` of the ``Ns`` system qubits (``features_Z``).

For each run the principal axes are fitted on the training samples of the
run (the first ``train_size`` samples, the split of the ridge readout), with
the bias column dropped, and the test samples are projected on them
(``--samples all`` draws every sample). The features are not standardized:
``M`` and ``<Z_k>`` both lie in ``[-1, 1]``, and the PCA sees them in the
units the ridge readout sees them in. Each panel is titled with the ridge
accuracy of that feature set in the run (``accuracy_M``, ``accuracy_M_Z``,
``accuracy``) and labels its axes with their share of the training
variance.

The figure ``pca_feature_spaces.png`` is a grid, one row per feature space
and one column per ``p``, written with its data
(:func:`QELM.utils.plots.save_figure`): the projected samples of each panel,
the variances and the rank of its PCA. ``--components 3`` draws the first
three principal components in 3D instead of two.

The sweep must hold one value of ``TFIM.h``, ``evolution.t`` and
``final_measurement`` for that ``N``, or the one to plot is chosen with
``--h``, ``--t`` and ``--final``. A point with several seeds draws the
smallest seed. For the figure of issue #225:

    N_VALUES=8 P_VALUES="0.0 0.5 1.0" DELTA=200 STATE=vector MAX_JOBS=3 \\
        ./experiments/QELM_Q/exp_QELM_Q.sh
    uv run python experiments/QELM_Q/plot_pca.py outputs/exp_QELM_Q/<date_time>
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from QELM.evaluation.metrics import accuracy
from QELM.evaluation.pca import TrainPCA, fit_pca
from QELM.training import RidgeReadout
from QELM.utils import latest_sweep, load_sweep
from QELM.utils.plots import COLORS, save_figure, subplots

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
# Feature spaces, one row each: name in the .npz -> (label, accuracy metric).
SPACES = {
    "M": (r"$[M]$", "accuracy_M"),
    "M_Z": (r"$[M, Z]$", "accuracy_M_Z"),
    "Z": (r"$[Z]$", "accuracy"),
}
NPZ = "exp_QELM_Q.npz"
FIGURE = "pca_feature_spaces"
SUM_FIGURE = "record_sum_direction"
RIDGE_FIGURE = "ridge_outputs"
# Feature spaces that hold the record, first in the features (issue #179).
RECORD_SPACES = ("M", "M_Z")


def select_runs(
    rows: Sequence[dict[str, Any]], N: int, **fixed: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the runs of size ``N`` with the given ``h``, ``t``, ``final``.

    Args:
        rows: Output of :func:`QELM.utils.load_sweep` with :data:`KEYS`.
        N: Size to keep.
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
        if int(r["N"]) == N
        and all(v is None or r[k] == v for k, v in fixed.items() if k in FIXED)
    ]
    if not kept:
        raise ValueError(f"No run with N = {N} and {fixed}.")
    values: dict[str, Any] = {"N": N}
    for key in FIXED:
        found = sorted({r[key] for r in kept}, key=str)
        if len(found) > 1:
            raise ValueError(f"The sweep has several {key} = {found}: pick one.")
        values[key] = found[0]
    return kept, values


def project_run(
    run_dir: Path, space: str, n_components: int = 2, samples: str = "test"
) -> tuple[np.ndarray, np.ndarray, TrainPCA]:
    """Fit the PCA of one feature space on the training samples of a run.

    Args:
        run_dir: Folder of one run, holding :data:`NPZ`.
        space: A key of :data:`SPACES`.
        n_components: Number of principal components.
        samples: ``"test"`` to project the test samples, ``"all"`` for every
            sample.

    Returns:
        The projected samples ``(n, n_components)``, their charges ``(n,)``
        and the PCA.
    """
    with np.load(run_dir / NPZ) as data:
        # The bias column (last) is constant: no variance, dropped.
        X, Y = data[f"features_{space}"][:, :-1], data["Y"]
        n_test = len(data["y_test"])
    train = len(Y) - n_test
    pca = fit_pca(X[:train], n_components)
    shown = slice(train, None) if samples == "test" else slice(None)
    return pca.transform(X[shown]), Y[shown], pca


def _axis_label(pca: TrainPCA, k: int) -> str:
    if pca.variances[k] == 0.0:
        return f"PC {k + 1} (no variance)"
    return f"PC {k + 1} ({100 * pca.explained[k]:.0f} %)"


def draw_panel(
    ax: Any, projected: np.ndarray, y: np.ndarray, pca: TrainPCA
) -> dict[str, Any]:
    """Draw the projected samples of one panel, colored by charge sector.

    Returns:
        The data of the panel: the charges, the coordinates of each sector,
        the variances, explained shares and rank of the PCA.
    """
    n_components = projected.shape[1]
    sectors = {}
    for k, charge in enumerate(np.unique(y)):
        points = projected[y == charge]
        coords = [points[:, c] for c in range(n_components)]
        ax.plot(
            *coords,
            ls="none",
            marker=".",
            ms=4,
            alpha=0.5,
            color=COLORS[k % len(COLORS)],
            label=f"Q = {charge:g}",
        )
        sectors[f"Q{charge:g}"] = points.tolist()
    ax.set_xlabel(_axis_label(pca, 0))
    ax.set_ylabel(_axis_label(pca, 1) if n_components > 1 else "")
    if n_components > 2:
        ax.set_zlabel(_axis_label(pca, 2))
    if pca.degenerate:
        # 3D axes place text in data coordinates unless text2D is used.
        text = ax.text2D if n_components > 2 else ax.text
        text(
            0.5,
            0.92,
            "zero variance: every sample at the origin",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    return {
        "sectors": sectors,
        "variances": pca.variances.tolist(),
        "explained": pca.explained.tolist(),
        "total_variance": pca.total_variance,
        "rank": pca.rank,
        "degenerate": pca.degenerate,
    }


def _grid(n_rows: int, n_cols: int, n_components: int) -> tuple[Figure, np.ndarray]:
    """Return the figure and its axes, 2D or 3D, shape ``(n_rows, n_cols)``."""
    size = (4.2 * n_cols, 3.8 * n_rows)
    if n_components < 3:
        return subplots(n_rows, n_cols, figsize=size, squeeze=False)
    fig, axes = subplots(
        n_rows,
        n_cols,
        figsize=size,
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    return fig, axes


def draw_grid(
    sweep_dir: Path,
    rows: Sequence[dict[str, Any]],
    values: dict[str, Any],
    p_values: Sequence[float] | None = None,
    n_components: int = 2,
    samples: str = "test",
) -> tuple[Figure, dict[str, Any]]:
    """Draw the PCA of each feature space (rows) for each ``p`` (columns).

    Args:
        sweep_dir: Folder of the sweep.
        rows: Runs of one ``(N, h, t, final)``, from :func:`select_runs`.
        values: Their ``N``, ``h``, ``t`` and ``final``.
        p_values: Values of ``p`` to draw, default every value of the sweep.
        n_components: 2 for a plane, 3 for a 3D view.
        samples: As in :func:`project_run`.

    Returns:
        The open figure and, per panel ``"<space>_p<p>"``, the run drawn,
        its ridge accuracy on that feature set and what :func:`draw_panel`
        returns.
    """
    ps = sorted({float(r["p"]) for r in rows} if p_values is None else p_values)
    fig, axes = _grid(len(SPACES), len(ps), n_components)
    extra: dict[str, Any] = {}
    for i, (space, (label, metric)) in enumerate(SPACES.items()):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            runs = sorted(
                (r for r in rows if np.isclose(r["p"], p)), key=lambda r: r["seed"]
            )
            if not runs:
                ax.set_axis_off()
                continue
            run = runs[0]
            projected, y, pca = project_run(
                sweep_dir / run["run"], space, n_components, samples
            )
            drawn = draw_panel(ax, projected, y, pca)
            score = run.get(metric)
            accuracy = "" if score is None else f", ridge accuracy {score:.2f}"
            ax.set_title(f"{label}, p = {p:g}{accuracy}", fontsize=10)
            extra[f"{space}_p{p:g}"] = {"run": run["run"], metric: score, **drawn}
    axes[0, -1].legend(fontsize=8)
    final = ", last round after the last step" if values["final"] else ""
    shown = "test samples" if samples == "test" else "all samples"
    fig.suptitle(
        f"PCA of the feature spaces (fitted on the training set, {shown}), "
        f"N = {values['N']}, h = {values['h']:g} J, t = {values['t']:g} / J{final}"
    )
    return fig, extra


def sum_direction(n_features: int, n_record: int) -> np.ndarray:
    """Return the unit vector that sums the record entries of a feature vector.

    The record ``M`` comes first in ``features_M`` and ``features_M_Z``, so
    ``x @ u`` is ``sum(M) / sqrt(n_record)``: the sum of the outcomes over
    every round and every qubit. At ``p = 1`` every qubit is measured in every
    round, and the outcomes of a round sum to ``Ns - 2 Q`` whatever the
    disorder: ``x @ u`` then takes one value per charge sector.

    Args:
        n_features: Length of a feature vector, bias column excluded.
        n_record: Number of record entries, ``n_rounds * Ns``.

    Returns:
        ``u``, shape ``(n_features,)``, ``1 / sqrt(n_record)`` on the record
        entries and 0 elsewhere.
    """
    if not 0 < n_record <= n_features:
        raise ValueError(f"Need 0 < n_record <= {n_features}, got {n_record}.")
    u = np.zeros(n_features)
    u[:n_record] = 1.0 / np.sqrt(n_record)
    return u


def refit_ridge(run_dir: Path, space: str) -> dict[str, Any]:
    """Refit the ridge readout of one feature set of a run on its split.

    ``exp_QELM_Q.npz`` keeps the weights of the ``<Z_k>`` readout only; the
    readouts of the other sets are refitted here, with the ``alpha`` of the
    run's config, the same way ``exp_QELM_Q.run`` fits them.

    Args:
        run_dir: Folder of one run, holding :data:`NPZ` and
            ``.hydra/config.yaml``.
        space: A key of :data:`SPACES`.

    Returns:
        ``W`` (bias last), the training and test features ``X_train``,
        ``X_test`` (bias dropped), the charges ``y_train``, ``y_test``, the
        test outputs ``output = phi @ W``, the ``threshold`` (midpoint of the
        two charges), the test ``accuracy`` and ``n_record`` (entries of the
        record ``M``, 0 for ``Z``).
    """
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with np.load(run_dir / NPZ) as data:
        phi, Y = data[f"features_{space}"], data["Y"]
        n_test = len(data["y_test"])
        n_record = int(np.prod(data["M"].shape[1:])) if space != "Z" else 0
    train = len(Y) - n_test
    readout = RidgeReadout.from_config(cfg)  # type: ignore[arg-type]
    readout.fit(phi[:train], Y[:train])
    assert readout.W is not None
    output = phi[train:] @ readout.W
    threshold = float(np.mean(np.unique(Y)))
    return {
        "W": np.asarray(readout.W),
        "X_train": phi[:train, :-1],
        "X_test": phi[train:, :-1],
        "y_train": Y[:train],
        "y_test": Y[train:],
        "output": output,
        "threshold": threshold,
        "accuracy": accuracy(Y[train:], output, threshold),
        "n_record": n_record,
    }


def _runs_by_p(
    rows: Sequence[dict[str, Any]], p_values: Sequence[float] | None
) -> list[tuple[float, dict[str, Any] | None]]:
    """Return, for each ``p``, the run of smallest seed (``None`` if none)."""
    ps = sorted({float(r["p"]) for r in rows} if p_values is None else p_values)
    out = []
    for p in ps:
        runs = sorted(
            (r for r in rows if np.isclose(r["p"], p)), key=lambda r: r["seed"]
        )
        out.append((p, runs[0] if runs else None))
    return out


def draw_sum_direction(
    sweep_dir: Path,
    rows: Sequence[dict[str, Any]],
    values: dict[str, Any],
    p_values: Sequence[float] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Draw the test samples on the plane (sum of the record, PC 1).

    One row per feature set of :data:`RECORD_SPACES`, one column per ``p``.
    The horizontal axis is the projection on :func:`sum_direction`, centered
    on the training mean; the vertical axis is the first principal component
    of the training samples once their part along ``u`` is removed. Each
    panel gives the share of the training variance along ``u`` (small: the
    PCA does not see it), the cosine between ``u`` and the ridge weights
    (bias dropped), and the ridge accuracy.

    Returns:
        The open figure and, per panel ``"<space>_p<p>"``, those numbers and
        the coordinates of each sector.
    """
    by_p = _runs_by_p(rows, p_values)
    fig, axes = subplots(
        len(RECORD_SPACES),
        len(by_p),
        figsize=(4.2 * len(by_p), 3.8 * len(RECORD_SPACES)),
        squeeze=False,
    )
    extra: dict[str, Any] = {}
    for i, space in enumerate(RECORD_SPACES):
        for j, (p, run) in enumerate(by_p):
            ax = axes[i, j]
            if run is None:
                ax.set_axis_off()
                continue
            fit = refit_ridge(sweep_dir / run["run"], space)
            X_train, X_test = fit["X_train"], fit["X_test"]
            u = sum_direction(X_train.shape[1], fit["n_record"])
            center = X_train.mean(axis=0)
            along = (X_test - center) @ u
            # PC 1 of what is left once the part along u is removed.
            rest = fit_pca((X_train - center) - np.outer((X_train - center) @ u, u), 1)
            across = rest.transform((X_test - center) - np.outer(along, u))
            total = float(np.sum(np.var(X_train, axis=0)))
            share = float(np.var(X_train @ u)) / total if total > 0 else 0.0
            w = fit["W"][:-1]
            norm = float(np.linalg.norm(w))
            cosine = float(w @ u) / norm if norm > 0 else 0.0
            sectors = {}
            for k, charge in enumerate(np.unique(fit["y_test"])):
                mask = fit["y_test"] == charge
                ax.plot(
                    along[mask],
                    across[mask, 0],
                    ls="none",
                    marker=".",
                    ms=4,
                    alpha=0.5,
                    color=COLORS[k % len(COLORS)],
                    label=f"Q = {charge:g}",
                )
                sectors[f"Q{charge:g}"] = np.column_stack(
                    [along[mask], across[mask, 0]]
                ).tolist()
            ax.set_xlabel(rf"record sum $u$ ({100 * share:.1f} % of var.)")
            ax.set_ylabel(_axis_label(rest, 0).replace("PC 1", r"PC 1 $\perp u$"))
            label = SPACES[space][0]
            ax.set_title(
                f"{label}, p = {p:g}\nridge accuracy {fit['accuracy']:.2f}, "
                f"cos(W, u) = {cosine:.2f}",
                fontsize=10,
            )
            extra[f"{space}_p{p:g}"] = {
                "run": run["run"],
                "accuracy": fit["accuracy"],
                "variance_share_u": share,
                "cos_W_u": cosine,
                "sectors": sectors,
            }
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(
        "Test samples along the sum of the measurement record, "
        f"N = {values['N']}, h = {values['h']:g} J, t = {values['t']:g} / J"
    )
    return fig, extra


def draw_ridge_outputs(
    sweep_dir: Path,
    rows: Sequence[dict[str, Any]],
    values: dict[str, Any],
    p_values: Sequence[float] | None = None,
    bins: int = 40,
) -> tuple[Figure, dict[str, Any]]:
    """Draw the histogram of the ridge output on the test samples, by sector.

    One row per feature space of :data:`SPACES`, one column per ``p``. The
    readout predicts the larger charge at or above the threshold (dashed):
    the accuracy is the share of each sector on its side of it, so two
    histograms on both sides of the line are what a ridge accuracy of 1
    means, whatever the PCA plane shows.

    Returns:
        The open figure and, per panel ``"<space>_p<p>"``, the threshold, the
        accuracy and the outputs of each sector.
    """
    by_p = _runs_by_p(rows, p_values)
    fig, axes = subplots(
        len(SPACES),
        len(by_p),
        figsize=(4.2 * len(by_p), 3.4 * len(SPACES)),
        squeeze=False,
    )
    extra: dict[str, Any] = {}
    for i, (space, (label, _)) in enumerate(SPACES.items()):
        for j, (p, run) in enumerate(by_p):
            ax = axes[i, j]
            if run is None:
                ax.set_axis_off()
                continue
            fit = refit_ridge(sweep_dir / run["run"], space)
            output, y = fit["output"], fit["y_test"]
            edges = np.histogram_bin_edges(output, bins=bins)
            outputs = {}
            for k, charge in enumerate(np.unique(y)):
                ax.hist(
                    output[y == charge],
                    bins=edges,
                    alpha=0.6,
                    color=COLORS[k % len(COLORS)],
                    label=f"Q = {charge:g}",
                )
                outputs[f"Q{charge:g}"] = output[y == charge].tolist()
            ax.axvline(fit["threshold"], color="gray", ls="--", lw=1.5)
            ax.set(xlabel=r"ridge output $\phi \cdot W$", ylabel="test samples")
            ax.set_title(
                f"{label}, p = {p:g}, ridge accuracy {fit['accuracy']:.2f}",
                fontsize=10,
            )
            extra[f"{space}_p{p:g}"] = {
                "run": run["run"],
                "threshold": fit["threshold"],
                "accuracy": fit["accuracy"],
                "outputs": outputs,
            }
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(
        "Ridge output on the test samples (dashed: decision threshold), "
        f"N = {values['N']}, h = {values['h']:g} J, t = {values['t']:g} / J"
    )
    return fig, extra


def plot_sweep(
    sweep_dir: Path,
    N: int = 8,
    p_values: Sequence[float] | None = None,
    n_components: int = 2,
    samples: str = "test",
    **fixed: Any,
) -> dict[str, Path]:
    """Write the figures of the sweep (``.png`` and ``.json``) in its folder.

    Args:
        sweep_dir: Folder of the sweep.
        N: Size to draw.
        p_values, n_components, samples: As in :func:`draw_grid`.
        fixed: ``h``, ``t``, ``final`` to keep, as in :func:`select_runs`.

    Returns:
        The path of each PNG file: ``"pca"`` (:data:`FIGURE`, ``_3d`` with
        three components), ``"sum"`` (:data:`SUM_FIGURE`,
        :func:`draw_sum_direction`) and ``"ridge"`` (:data:`RIDGE_FIGURE`,
        :func:`draw_ridge_outputs`).
    """
    rows, values = select_runs(load_sweep(sweep_dir, keys=KEYS), N, **fixed)
    out = {}
    fig, extra = draw_grid(sweep_dir, rows, values, p_values, n_components, samples)
    name = FIGURE if n_components < 3 else f"{FIGURE}_3d"
    out["pca"] = save_figure(fig, sweep_dir, name, data={**values, "panels": extra})
    plt.close(fig)
    for key, draw, name in (
        ("sum", draw_sum_direction, SUM_FIGURE),
        ("ridge", draw_ridge_outputs, RIDGE_FIGURE),
    ):
        fig, extra = draw(sweep_dir, rows, values, p_values)
        out[key] = save_figure(fig, sweep_dir, name, data={**values, "panels": extra})
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
    parser.add_argument("--N", type=int, default=8, help="size N to plot")
    parser.add_argument("--h", type=float, default=None, help="TFIM.h to plot")
    parser.add_argument("--t", type=float, default=None, help="evolution.t to plot")
    parser.add_argument(
        "--final",
        choices=("true", "false"),
        default=None,
        help="final_measurement to plot",
    )
    parser.add_argument(
        "--p", type=float, nargs="+", default=None, help="p of the columns"
    )
    parser.add_argument(
        "--components",
        type=int,
        choices=(2, 3),
        default=2,
        help="principal components drawn (3 for a 3D view)",
    )
    parser.add_argument(
        "--samples",
        choices=("test", "all"),
        default="test",
        help="samples projected on the training PCA",
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "exp_QELM_Q")
    final = None if args.final is None else args.final == "true"
    pngs = plot_sweep(
        sweep_dir,
        N=args.N,
        p_values=args.p,
        n_components=args.components,
        samples=args.samples,
        h=args.h,
        t=args.t,
        final=final,
    )
    for png in pngs.values():
        print(f"Figure: {png}")


if __name__ == "__main__":
    main()
