#!/usr/bin/env python3
"""Table (PNG) of p_c and nu from Delta, S(N/2) and I3, with the literature (#358).

Reads the collapse summaries of the purification entropy ``Delta`` (the
Renyi-2 entropy of the references, #346 Clifford and #348 Haar) and of the
bipartite entropy (``analyse.py`` of this folder), and draws one table:
rows Clifford / Haar x (``p_c``, ``nu``), columns ``Delta``, ``S(N/2)``,
``I3`` and the literature. Each cell gives the best collapse with its
bootstrap error, and below it the range over the data variants (systematic
spread).

    uv run python experiments/bipartite_entropy/summary_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from QELM.utils.plots import save_figure, subplots

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LITERATURE = {
    "clifford": ("0.1598", "1.30", "Gullans & Huse,\nPRX 10, 041020 (2020)"),
    "haar": ("0.168(5)", "1.2(2)", "Zabalo et al.,\nPRB 101, 060301 (2020)"),
}
# Haar S(N/2): flat valley of the quality map, no constraint (README).
HAAR_S_RANGE = {"p_c": "0.15 - 0.25", "nu": "0.6 - 1.0"}


def _value(fit: dict[str, Any], key: str) -> str:
    """``0.1590(3)`` style: value with the bootstrap error on its last digits."""
    v, e = fit[key], fit[f"{key}_err"]
    digits = 4 if key == "p_c" else 2
    err = round(e * 10**digits)
    return f"{v:.{digits}f}({max(err, 1)})"


def _range(fits: list[dict[str, Any]], key: str) -> str:
    digits = 3 if key == "p_c" else 2
    vals = [f[key] for f in fits]
    return f"[{min(vals):.{digits}f} - {max(vals):.{digits}f}]"


def _block(summary: dict[str, Any], main: str, variants: list[str]) -> dict[str, str]:
    fits = [summary[main]["fit"]] + [
        summary[v]["fit"] for v in variants if v in summary
    ]
    return {
        key: f"{_value(summary[main]['fit'], key)}\n{_range(fits, key)}"
        for key in ("p_c", "nu")
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args(argv)

    bip = json.loads((HERE / "figures" / "summary.json").read_text())
    delta = {
        "clifford": json.loads(
            (EXP / "clifford_collapse_map" / "figures" / "summary.json").read_text()
        ),
        "haar": json.loads(
            (EXP / "haar_collapse_map" / "figures" / "summary.json").read_text()
        ),
    }
    cells: dict[str, dict[str, dict[str, str]]] = {}
    sizes: dict[str, dict[str, str]] = {}
    cells["clifford"] = {
        "Delta": _block(delta["clifford"], "main", ["no_N16", "log", "window"]),
        "S": _block(bip["clifford"], "S1", ["S1_drop"]),
        "I3": _block(bip["clifford"], "I3", ["I3_drop", "I3_window"]),
    }
    sizes["clifford"] = {"Delta": "N = 16-128", "S": "N = 16-256", "I3": "N = 16-256"}
    haar_s = bip["haar"]["S1"]["fit"]
    cells["haar"] = {
        "Delta": _block(delta["haar"], "main", ["no_N4", "log", "window"]),
        "S": {
            k: f"not constrained\n(flat valley {HAAR_S_RANGE[k]};\n"
            f"min. {haar_s[k]:.{3 if k == 'p_c' else 2}f})"
            for k in ("p_c", "nu")
        },
        "I3": _block(bip["haar"], "I3", ["I3_2"]),
    }
    sizes["haar"] = {"Delta": "N = 4-12", "S": "N = 8-20", "I3": "N = 8, 12, 16, 20"}

    header = [
        "",
        "",
        "Purification\nDelta (Renyi-2 of the refs)",
        "Half-chain\nS(N/2) - S(p_c, N)",
        "Tripartite\ninformation I3",
        "Literature",
    ]
    rows: list[list[str]] = []
    for model, name in (("clifford", "Clifford"), ("haar", "Haar")):
        lit = LITERATURE[model]
        for k, (key, label) in enumerate((("p_c", "p_c"), ("nu", "nu"))):
            rows.append(
                [
                    f"{name}\n"
                    + "\n".join(f"{m}: {sizes[model][m]}" for m in cells[model])
                    if k == 0
                    else "",
                    label,
                    cells[model]["Delta"][key],
                    cells[model]["S"][key],
                    cells[model]["I3"][key],
                    lit[k] + (f"\n{lit[2]}" if k == 0 else ""),
                ]
            )

    fig, ax = subplots(figsize=(15, 6.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="center",
        colWidths=[0.17, 0.05, 0.19, 0.2, 0.19, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_height(0.2 if r else 0.13)
        if r == 0:
            cell.set_facecolor("#dfe7f2")
            cell.set_text_props(weight="bold")
        elif c in (0, 1):
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold" if c == 1 else "normal", fontsize=8.5)
        elif c == 5:
            cell.set_facecolor("#fff4e5")
        if c == 0 and r in (1, 3):
            cell.visible_edges = "LTR"
        if c == 0 and r in (2, 4):
            cell.visible_edges = "LBR"
            cell.set_facecolor("#f2f2f2")
        if r > 0 and c == 3 and rows[r - 1][3].startswith("not"):
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(color="0.35", style="italic")
    ax.set_title(
        "Critical point p_c and exponent nu of the MIPT from three collapses "
        "(periodic brickwork, T = 2N)\n"
        "value(bootstrap error) of the main collapse; [range over data variants] "
        "= systematic spread",
        fontsize=10,
    )
    save_figure(fig, args.out, "summary_table", data={"cells": cells, "sizes": sizes})


if __name__ == "__main__":
    main()
