#!/usr/bin/env python3
"""Figures of a sweep of ``haar_readouts.sh``: one figure per readout.

Reads every run folder of the sweep (one per circuit, gate draw, ``N`` and
seed),
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

When the sweep holds several circuits (Haar, Clifford), each gets its own
figures, ``<circuit>_<readout>.png``. When the runs score the ridge on the
record ``M``, on the observables of the final state (the QELM, set ``all``)
and on both (the QELM-M, set ``M_all``), ``compare_<circuit>.png`` draws the
three on the same panels: one color per feature set, solid lines for a fixed
circuit and dashed lines for new gates per trajectory.

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

KEYS = (("N", "N"), ("seed", "seed"), ("models", "models"))
# Circuit names in the figures, by models.
CIRCUIT_TITLES = {"haar": "Haar", "clifford": "Clifford"}
# Panel titles of the readouts.
TITLES = {
    "ridge": "Ridge (linear)",
    "ridge_pairs": "Ridge on outcomes and their pairwise products",
    "knn": "k nearest neighbors",
    "mlp": "Neural network (MLP 64-64)",
    "gradient_boosting": "Gradient-boosted trees",
}


# Feature sets other than the record, as written in the panel titles.
FEATURE_LABELS = {
    "X": "<X_k>",
    "Y": "<Y_k>",
    "Z": "<Z_k>",
    "XX": "<X_i X_j>",
    "YY": "<Y_i Y_j>",
    "ZZ": "<Z_i Z_j>",
    "all": "all six observable sets",
    "M_all": "M and all six observable sets",
}


def title(name: str) -> str:
    """Panel title of a readout of the record (``ridge``) or of a feature set
    of the final state (``ridge_ZZ``)."""
    if name in TITLES:
        return TITLES[name]
    for feature, label in FEATURE_LABELS.items():
        readout = name.removesuffix(f"_{feature}")
        if readout != name and readout in TITLES:
            return f"{TITLES[readout]} on {label}"
    return name


def load_rows(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return one row per run, with its circuit, gate draw and ``p`` values.

    Raises:
        ValueError: If the folder holds no run.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    if not rows:
        raise ValueError(f"No run in {sweep_dir}.")
    for row in rows:
        cfg = OmegaConf.load(Path(sweep_dir) / row["run"] / ".hydra" / "config.yaml")
        row["p_values"] = [float(v) for v in cfg.readouts_task.p_values]
        model = str(row["models"] or "haar_circuit")
        row["circuit"] = model.removesuffix("_circuit")
        row["gates"] = "fixed" if cfg[model].get("fixed_instance") else "random"
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
    order = [*TITLES] + [f"{r}_{f}" for r in TITLES for f in FEATURE_LABELS]
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


def _three_panels(
    lines: list[tuple[str, list[dict[str, Any]], str, dict[str, Any]]],
    title_text: str,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw ``S_R``, accuracy and ``R2`` against ``p``.

    Args:
        lines: One ``(label, runs, score, style)`` per curve: the runs
            averaged over, the score name (``ridge``, ``ridge_all``...) and
            the matplotlib style. The purification panel gets one curve per
            distinct set of runs.
        title_text: Title of the accuracy and ``R2`` panels.
    """
    fig, (ax_s, ax_a, ax_r) = subplots(1, 3, figsize=(15.6, 4.2))
    table: dict[str, Any] = {}
    drawn_s: set[str] = set()
    for label, runs, score, style in lines:
        panels = [(ax_a, f"accuracy_{score}"), (ax_r, f"r2_{score}")]
        key_s = style.get("s_label", label)
        if key_s not in drawn_s:
            drawn_s.add(key_s)
            panels.insert(0, (ax_s, "S_R_mean"))
        for ax, name in panels:
            ps, y, e = curve(runs, name)
            shown = key_s if ax is ax_s else label
            plot_style = {k: v for k, v in style.items() if k != "s_label"}
            if ax is ax_s:
                plot_style["color"] = style.get("s_color", plot_style.get("color"))
                plot_style.pop("s_color", None)
            else:
                plot_style.pop("s_color", None)
            ax.errorbar(ps, y, yerr=e, marker="o", ms=4, label=shown, **plot_style)
            table[f"{shown}: {name}"] = {
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
    ax_a.set_title(title_text)
    ax_r.axhline(0.0, color="0.6", lw=1)
    # Overfitted outputs can fall well below 0: keep every point in view.
    r2_low = min(
        (min(v["mean"]) for k, v in table.items() if ": r2_" in k),
        default=0.0,
    )
    ax_r.set_ylim(min(-0.3, float(np.nan_to_num(r2_low)) - 0.05), 1.02)
    ax_r.set_ylabel(r"$R^2$ of the output (test)")
    ax_r.set_title(title_text)
    for ax in (ax_s, ax_a, ax_r):
        ax.set_xlabel("p")
        ax.legend(frameon=False, fontsize=9)
    return fig, table


def _by_run(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group the runs by circuit, gate draw and ``N`` (over the seeds)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = f"{row['circuit']}, {row['gates']}, N = {int(row['N'])}"
        groups.setdefault(label, []).append(row)
    return dict(sorted(groups.items()))


def draw_readout(
    rows: list[dict[str, Any]], readout: str
) -> tuple[plt.Figure, dict[str, Any]]:
    """Purification curve, accuracy and ``R2`` of one readout against ``p``."""
    groups = _by_run(rows)
    colors = color_map(list(groups))
    lines = [
        (label, runs, readout, {"color": colors[label]})
        for label, runs in groups.items()
    ]
    return _three_panels(lines, title(readout))


def draw_comparison(
    rows: list[dict[str, Any]], circuit: str, readout: str = "ridge"
) -> tuple[plt.Figure, dict[str, Any]] | None:
    """One circuit: ``readout`` on the record (``M``), on the observables of
    the final state (QELM, set ``all``) and on both (QELM-M, ``M_all``).

    Colors are the feature sets; solid lines a fixed circuit, dashed lines
    new gates per trajectory. ``None`` if the runs lack these scores.
    """
    runs_c = [r for r in rows if r["circuit"] == circuit]
    groups = _by_run(runs_c)
    scores = {f"{readout}": "M", f"{readout}_all": "QELM", f"{readout}_M_all": "QELM-M"}
    present = [s for s in scores if any(f"accuracy_{s}_p0" in r for r in runs_c)]
    if not present:
        return None
    colors = color_map([scores[s] for s in present])
    gate_colors = color_map(list(groups))
    lines = []
    for label, runs in groups.items():
        dashed = runs[0]["gates"] == "random"
        for s in present:
            style = {
                "color": colors[scores[s]],
                "ls": "--" if dashed else "-",
                "s_label": label,
                "s_color": gate_colors[label],
            }
            lines.append((f"{scores[s]} ({runs[0]['gates']} gates)", runs, s, style))
    return _three_panels(
        lines, f"{CIRCUIT_TITLES.get(circuit, circuit)}: {title(readout)}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="folder of one haar_readouts.sh sweep")
    parser.add_argument("--out", type=Path, default=None, help="figure folder")
    args = parser.parse_args(argv)
    out = args.out or args.sweep
    rows = load_rows(args.sweep)
    many = len({row["circuit"] for row in rows}) > 1
    for circuit in sorted({row["circuit"] for row in rows}):
        rows_c = [row for row in rows if row["circuit"] == circuit]
        for readout in readouts(rows_c):
            fig, table = draw_readout(rows_c, readout)
            save_figure(
                fig, out, f"{circuit}_{readout}" if many else readout, data=table
            )
            plt.close(fig)
        compared = draw_comparison(rows, circuit)
        if compared is not None:
            fig, table = compared
            save_figure(fig, out, f"compare_{circuit}", data=table)
            plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
