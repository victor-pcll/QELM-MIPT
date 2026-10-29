#!/usr/bin/env python3
"""K-NN readouts of a ``exp_QELM_Q.sh`` sweep against ``p`` (issue #214).

Reads the run folders of one sweep of ``experiments/QELM_Q/exp_QELM_Q.sh``
and, for each run, fits K-NN classifiers of ``Q`` on the same split and the
same features as the ridge readout: the final ``<Z_k>`` of ``X`` without the
bias column, the first ``samples.train_size`` samples to train, the rest to
test. Every accuracy is averaged over the seeds, with the standard error over
the seeds as error bars. One column of panels per ``N``, and one row per
value of ``projective_measurement.final_measurement`` found in the sweep.

Two figures, each with its data (:func:`QELM.utils.plots.save_figure`), in
the sweep folder:

- ``knn_parity_vs_p.png``: the accuracy of scikit-learn's
  ``KNeighborsClassifier(n_neighbors=round(sqrt(n_train)), weights="distance")``
  (solid, the K-NN of the Bayes ceilings) and of :class:`CustomKNN` with the
  same ``K`` (dashed). Each panel prints the fraction of test points that the
  two classify the same way and the largest gap between the two accuracy
  curves. With ``final_measurement=false`` the ``<Z_k>`` are continuous and
  the two are identical. With ``final_measurement=true`` the measured qubits
  sit at ``±1``, so at large ``p`` many training points are at the same
  distance from a test point; scikit-learn then breaks the tie in an order
  set by its search algorithm, :class:`CustomKNN` averages the vote over all
  tie breaks (``ties="average"``), and the panel gives the fraction of test
  points with a tie at the ``K``-th neighbor.
- ``knn_K_vs_p.png``: the accuracy of :class:`CustomKNN` for several ``K``
  (``--K``, default ``1``, ``5``, ``round(sqrt(n_train))`` and three times
  that), with the ridge readout of the run (``accuracy`` of ``metrics.json``)
  in gray, dashed. A small ``K`` follows small clusters of samples; a large
  ``K`` averages over a wider neighborhood, and merges clusters of the two
  sectors that lie close together.

The accuracies of every run are in ``knn_runs.json``.

    ./experiments/QELM_Q/exp_QELM_Q.sh   # with P_VALUES, SEEDS, N_VALUES...
    uv run python experiments/QELM_Q/plot_knn.py outputs/exp_QELM_Q/<date_time>
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from sklearn.neighbors import KNeighborsClassifier

from QELM.evaluation.knn import CustomKNN, default_n_neighbors
from QELM.utils import load_sweep, summarize
from QELM.utils.plots import (
    color_map,
    draw_curve,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

KEYS: tuple[tuple[str, str], ...] = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("final", "projective_measurement.final_measurement"),
    ("h", "TFIM.h"),
    ("train_size", "samples.train_size"),
)

# The default values of --K: "sqrt" is round(sqrt(n_train)), "3sqrt" three
# times that.
DEFAULT_K = ("1", "5", "sqrt", "3sqrt")
K_TOKEN = re.compile(r"^(\d*)sqrt$")


def resolve_k(token: str, n_train: int) -> int:
    """Return the ``K`` of a ``--K`` token for a training set of ``n_train``.

    Args:
        token: A positive integer, ``"sqrt"`` for ``round(sqrt(n_train))``,
            or ``"<m>sqrt"`` for ``m`` times that.
        n_train: Number of training samples.

    Returns:
        ``K``, at most ``n_train``.

    Raises:
        ValueError: If the token is not of one of these forms.
    """
    match = K_TOKEN.match(token)
    if match:
        factor = int(match.group(1) or 1)
        K = factor * default_n_neighbors(n_train)
    elif token.isdigit() and int(token) > 0:
        K = int(token)
    else:
        raise ValueError(f"K must be a positive integer or <m>sqrt, got {token!r}")
    return min(K, n_train)


def k_label(token: str, n_train: int) -> str:
    """Legend label of a ``--K`` token, with its value."""
    K = resolve_k(token, n_train)
    match = K_TOKEN.match(token)
    if not match:
        return f"K = {K}"
    factor = match.group(1)
    return f"K = {factor or ''}{'·' if factor else ''}round(√n) = {K}"


def tied_kth(X_train: np.ndarray, X_test: np.ndarray, K: int) -> np.ndarray:
    """Test points whose ``K``-th and ``(K+1)``-th distances are equal."""
    if K >= X_train.shape[0]:
        return np.zeros(X_test.shape[0], dtype=bool)
    d2 = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=-1)
    d2 = np.sort(d2, axis=1)
    return d2[:, K - 1] == d2[:, K]


def score_run(run_dir: Path, train_size: int, k_tokens: Sequence[str]) -> dict:
    """Fit the K-NN readouts of one run and return their accuracies.

    Args:
        run_dir: Run folder holding ``exp_QELM_Q.npz``.
        train_size: ``samples.train_size`` of the run.
        k_tokens: The ``--K`` tokens.

    Returns:
        ``accuracy_knn_sklearn``, ``accuracy_knn_custom``, ``agreement`` (the
        fraction of test points both classify the same way), ``tied`` (the
        fraction with a tie at the ``K``-th neighbor), ``K`` and
        ``accuracy_knn_<token>`` for each token.
    """
    with np.load(run_dir / "exp_QELM_Q.npz") as data:
        X, Y = data["X"][:, :-1], data["Y"]
    X_train, y_train = X[:train_size], Y[:train_size]
    X_test, y_test = X[train_size:], Y[train_size:]
    K = default_n_neighbors(train_size)
    ref = KNeighborsClassifier(n_neighbors=K, weights="distance").fit(X_train, y_train)
    ref_pred = ref.predict(X_test)
    pred = CustomKNN().fit(X_train, y_train).predict(X_test)
    out: dict[str, Any] = {
        "K": K,
        "accuracy_knn_sklearn": float(np.mean(ref_pred == y_test)),
        "accuracy_knn_custom": float(np.mean(pred == y_test)),
        "agreement": float(np.mean(ref_pred == pred)),
        "tied": float(np.mean(tied_kth(X_train, X_test, K))),
    }
    for token in k_tokens:
        model = CustomKNN(n_neighbors=resolve_k(token, train_size))
        out[f"accuracy_knn_{token}"] = model.fit(X_train, y_train).score(X_test, y_test)
    return out


def load_knn_sweep(
    sweep_dir: str | Path, k_tokens: Sequence[str] = DEFAULT_K
) -> list[dict[str, Any]]:
    """Return one row per run: its config keys, ``metrics.json`` and K-NN scores.

    Raises:
        FileNotFoundError: If ``sweep_dir`` holds no run.
        ValueError: If the runs have several fields ``TFIM.h`` (the panels
            would mix them).
    """
    rows = load_sweep(sweep_dir, KEYS)
    fields = {r["h"] for r in rows}
    if len(fields) > 1:
        raise ValueError(f"the sweep has several fields TFIM.h: {sorted(fields)}")
    for row in rows:
        row["final"] = bool(row["final"])
        row.update(
            score_run(Path(sweep_dir) / row["run"], int(row["train_size"]), k_tokens)
        )
    return rows


def _grid(rows: Sequence[dict[str, Any]]) -> tuple[list[bool], list[int], Figure, Any]:
    finals = sorted({r["final"] for r in rows})
    n_values = sizes(rows)
    fig, axes = subplots(
        len(finals),
        len(n_values),
        figsize=(5.5 * len(n_values), 4.2 * len(finals)),
        squeeze=False,
        sharey=True,
    )
    return finals, n_values, fig, axes


def _panel(ax: Any, N: int, h: Any, final: bool, n_finals: int) -> None:
    title = f"N = {N}, h = {h}"
    if n_finals > 1:
        title += ", final round" if final else ", no final round"
    ax.set_title(title)
    ax.set_xlabel("p")


def plot_parity(rows: Sequence[dict[str, Any]]) -> tuple[Figure, list[dict]]:
    """The scikit-learn and CustomKNN accuracies against ``p``.

    Returns:
        The figure and, per panel, the agreement, tie fraction and gap.
    """
    finals, n_values, fig, axes = _grid(rows)
    stats = []
    for i, final in enumerate(finals):
        for j, N in enumerate(n_values):
            ax = axes[i, j]
            sub = [r for r in rows if r["final"] == final and int(r["N"]) == N]
            summary = summarize(
                sub,
                by=("N", "p"),
                metrics=("accuracy_knn_sklearn", "accuracy_knn_custom"),
            )
            ref = sweep_curve(summary, N, "p", "accuracy_knn_sklearn")
            ours = sweep_curve(summary, N, "p", "accuracy_knn_custom")
            draw_curve(ax, ref, color="C0", label="scikit-learn")
            bars = draw_curve(ax, ours, color="C3", label="CustomKNN", linestyle="--")
            bars.lines[0].set_marker("x")
            agreement = float(np.mean([r["agreement"] for r in sub]))
            tied = float(np.mean([r["tied"] for r in sub]))
            gap = float(np.max(np.abs(ref.mean - ours.mean)))
            ax.text(
                0.03,
                0.04,
                f"same label: {100 * agreement:.2f} %\n"
                f"tie at K-th: {100 * tied:.1f} %\n"
                f"max |Δ acc| = {gap:.3f}",
                transform=ax.transAxes,
                fontsize=9,
            )
            _panel(ax, N, sub[0]["h"], final, len(finals))
            stats.append(
                {
                    "N": N,
                    "final": final,
                    "agreement": agreement,
                    "tied": tied,
                    "gap": gap,
                }
            )
    axes[0, 0].set_ylabel("K-NN accuracy")
    axes[0, 0].legend(loc="upper right")
    return fig, stats


def plot_k(rows: Sequence[dict[str, Any]], k_tokens: Sequence[str]) -> Figure:
    """The CustomKNN accuracy against ``p`` for each ``K``, and the ridge."""
    finals, n_values, fig, axes = _grid(rows)
    colors = color_map(k_tokens)
    metrics = (*(f"accuracy_knn_{t}" for t in k_tokens), "accuracy")
    for i, final in enumerate(finals):
        for j, N in enumerate(n_values):
            ax = axes[i, j]
            sub = [r for r in rows if r["final"] == final and int(r["N"]) == N]
            summary = summarize(sub, by=("N", "p"), metrics=metrics)
            n_train = int(sub[0]["train_size"])
            for token in k_tokens:
                curve = sweep_curve(summary, N, "p", f"accuracy_knn_{token}")
                draw_curve(ax, curve, colors[token], label=k_label(token, n_train))
            ridge = sweep_curve(summary, N, "p", "accuracy")
            draw_curve(ax, ridge, "0.4", label="ridge", linestyle="--")
            _panel(ax, N, sub[0]["h"], final, len(finals))
            ax.legend(loc="lower left", fontsize=8)
    axes[0, 0].set_ylabel("CustomKNN accuracy")
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path, help="folder of one sweep")
    parser.add_argument(
        "--K",
        nargs="+",
        default=list(DEFAULT_K),
        help="values of K: integers, sqrt for round(sqrt(n_train)), <m>sqrt",
    )
    args = parser.parse_args(argv)
    for token in args.K:
        resolve_k(token, 1)  # Reject a bad token before the slow part.

    rows = load_knn_sweep(args.sweep_dir, args.K)
    columns = (
        "run",
        "N",
        "seed",
        "p",
        "final",
        "train_size",
        "K",
        "accuracy",
        "accuracy_knn_sklearn",
        "accuracy_knn_custom",
        "agreement",
        "tied",
        *(f"accuracy_knn_{t}" for t in args.K),
    )
    table = [{c: r[c] for c in columns} for r in rows]
    (args.sweep_dir / "knn_runs.json").write_text(json.dumps(table, indent=2) + "\n")

    fig, stats = plot_parity(rows)
    print(save_figure(fig, args.sweep_dir, "knn_parity_vs_p", {"panels": stats}))
    for s in stats:
        print(
            f"N = {s['N']}, final = {s['final']}: same label "
            f"{100 * s['agreement']:.2f} %, tie {100 * s['tied']:.1f} %, "
            f"max gap {s['gap']:.4f}"
        )
    fig = plot_k(rows, args.K)
    print(save_figure(fig, args.sweep_dir, "knn_K_vs_p", {"K": list(args.K)}))


if __name__ == "__main__":
    main()
