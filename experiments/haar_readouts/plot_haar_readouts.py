#!/usr/bin/env python3
"""Figures of a sweep of ``haar_readouts.sh``: one figure per readout.

Reads every run folder of the sweep (one per gate draw, ``N`` and seed),
averages over the seeds (standard error as error bars) and writes, for each
readout of the runs, ``<readout>.png`` and its JSON
(:func:`QELM.utils.plots.save_figure`) in the sweep folder unless ``--out``
is given. Each figure has three panels against ``p``, one curve per gate draw
(new gates per trajectory, ``random``, or one fixed circuit, ``fixed``) and
``N``:

- left: the mean entropy of the reference qubit ``S_R`` (bits), the
  purification curve, 1 while the system hides the charge and 0 once the
  record fixes it. It does not depend on the readout;
- middle: the accuracy of the readout on the charge sector, on the test
  set, with the chance level 1/2 (grey line);
- right: the ``R2`` of the output of the readout against the label, on the
  test set, with 0 (grey line) for an output that ignores the record.

    uv run python experiments/haar_readouts/plot_haar_readouts.py \\
        outputs/haar_readouts/<date_time>
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from QELM.utils.plots import color_map, save_figure, subplots
from QELM.utils.sweep import load_sweep

KEYS = (("N", "N"), ("seed", "seed"), ("fixed", "haar_circuit.fixed_instance"))
# Panel titles of the readouts.
TITLES = {
    "ridge": "Ridge (linear)",
    "ridge_pairs": "Ridge on outcomes and their pairwise products",
    "knn": "k nearest neighbors",
    "mlp": "Neural network (MLP 64-64)",
    "gradient_boosting": "Gradient-boosted trees",
}


def load_rows(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return one row per run, with its folder, gate draw and ``p`` values.

    Raises:
        ValueError: If the folder holds no run.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    if not rows:
        raise ValueError(f"No run in {sweep_dir}.")
    for row in rows:
        cfg = OmegaConf.load(Path(sweep_dir) / row["run"] / ".hydra" / "config.yaml")
        row["p_values"] = [float(v) for v in cfg.readouts_task.p_values]
        row["gates"] = "fixed" if row["fixed"] else "random"
    return rows


def readouts(rows: list[dict[str, Any]]) -> list[str]:
    """Return the readouts scored in the runs, in the order of :data:`TITLES`."""
    found = {
        m.group(1)
        for row in rows
        for key in row
        if (m := re.fullmatch(r"accuracy_(.+)_p[0-9.e+-]+", key))
        and m.group(1) != "baseline"
    }
    order = list(TITLES)
    return sorted(
        found, key=lambda n: (order.index(n) if n in order else len(order), n)
    )


def curve(
    runs: list[dict[str, Any]], name: str
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return ``(p, mean, err)`` of the score ``<name>_p<p>`` over the seeds."""
    ps = runs[0]["p_values"]
    mean, err = [], []
    for p in ps:
        values = np.asarray(
            [r[f"{name}_p{p:g}"] for r in runs if r.get(f"{name}_p{p:g}") is not None],
            dtype=float,
        )
        mean.append(float(values.mean()) if values.size else float("nan"))
        err.append(
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
    return ps, np.array(mean), np.array(err)


def draw_readout(
    rows: list[dict[str, Any]], readout: str
) -> tuple[plt.Figure, dict[str, Any]]:
    """Purification curve, accuracy and ``R2`` of one readout against ``p``."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row['gates']}, N = {int(row['N'])}", []).append(row)
    labels = sorted(groups, key=lambda k: (not k.startswith("random"), k))
    colors = color_map(labels)
    fig, (ax_s, ax_a, ax_r) = subplots(1, 3, figsize=(15.6, 4.2))
    table: dict[str, Any] = {}
    for label in labels:
        runs = groups[label]
        for ax, name in (
            (ax_s, "S_R_mean"),
            (ax_a, f"accuracy_{readout}"),
            (ax_r, f"r2_{readout}"),
        ):
            ps, y, e = curve(runs, name)
            ax.errorbar(
                ps, y, yerr=e, color=colors[label], marker="o", ms=4, label=label
            )
            table[f"{label}: {name}"] = {
                "p": ps,
                "mean": y.tolist(),
                "err": e.tolist(),
                "n_seeds": len(runs),
            }
    ax_s.set_ylim(-0.03, 1.03)
    ax_s.set_ylabel(r"$\overline{S_R}$ (bits)")
    ax_s.set_title("Purification of the reference qubit")
    ax_a.axhline(0.5, color="0.6", lw=1)
    ax_a.set_ylim(0.4, 1.02)
    ax_a.set_ylabel("accuracy on the charge sector (test)")
    ax_a.set_title(TITLES.get(readout, readout))
    ax_r.axhline(0.0, color="0.6", lw=1)
    # Overfitted outputs can fall well below 0: keep every point in view.
    r2_low = min(
        (min(v["mean"]) for k, v in table.items() if k.endswith(f"r2_{readout}")),
        default=0.0,
    )
    ax_r.set_ylim(min(-0.3, float(np.nan_to_num(r2_low)) - 0.05), 1.02)
    ax_r.set_ylabel(r"$R^2$ of the output (test)")
    ax_r.set_title(TITLES.get(readout, readout))
    for ax in (ax_s, ax_a, ax_r):
        ax.set_xlabel("p")
        ax.legend(frameon=False, fontsize=9)
    return fig, table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="folder of one haar_readouts.sh sweep")
    parser.add_argument("--out", type=Path, default=None, help="figure folder")
    args = parser.parse_args(argv)
    out = args.out or args.sweep
    rows = load_rows(args.sweep)
    for readout in readouts(rows):
        fig, table = draw_readout(rows, readout)
        save_figure(fig, out, readout, data=table)
        plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
