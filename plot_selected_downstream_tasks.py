#!/usr/bin/env python3
"""Publication figure/table for validation-selected downstream tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = [
    ("strength_N100000", r"$10^5$-cycle strength", r"MAE ($\log_{10}$ MPa)"),
    ("life_at_mid_stress", "Life at service stress", r"MAE ($\log_{10}$ cycles)"),
    ("mean_log_life", "Mean curve life reserve", r"Integral MAE ($\log_{10}$ cycles)"),
    ("mission_balanced_log_damage", "Mission-spectrum Miner damage", r"MAE ($\log_{10}D$)"),
]
BASELINES = ["ExtraTrees", "PCHIP"]
COLORS = {"S-PA-CBB": "#315F98", "PA-CBB": "#CC6F76",
          "ExtraTrees": "#809BC4", "PCHIP": "#A9B8D5"}


def main(args):
    data = json.loads(args.metrics.read_text())
    lookup = {(r["split"], r["known"], r["task"], r["method"]): r for r in data}
    proposed = args.method_name
    methods = [proposed] + BASELINES
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), constrained_layout=True)
    width = 0.24
    for label, (task, title, ylabel) in zip("abcd", TASKS):
        ax = axes.flat["abcd".index(label)]
        x = np.arange(3)
        for offset, method in enumerate(methods):
            values = [lookup[("test", k, task, method)]["mae"] for k in (2, 3, 4)]
            ax.bar(x + (offset - 1) * width, values, width=width, color=COLORS[method],
                   edgecolor="white", linewidth=0.5, label=method)
        gains = []
        for k in (2, 3, 4):
            pa = lookup[("test", k, task, proposed)]["mae"]
            base = min(lookup[("test", k, task, m)]["mae"] for m in BASELINES)
            gains.append(100 * (base - pa) / base)
        ymax = max(p.get_height() for p in ax.patches)
        for i, gain in enumerate(gains):
            ax.text(i - width, lookup[("test", i + 2, task, proposed)]["mae"] + 0.025 * ymax,
                    f"{gain:.0f}%", ha="center", va="bottom", fontsize=7,
                    color="#9A414A", fontweight="bold")
        ax.set_xticks(x, ["2 tests", "3 tests", "4 tests"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"({label}) {title}", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(args.output / f"selected_downstream_tasks.{suffix}", dpi=400,
                    bbox_inches="tight")
    plt.close(fig)

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Validation-selected downstream tasks on the untouched test split. Gains are relative to the better of ExtraTrees and PCHIP under identical sparse observations. Basquin remains a physics ablation and is not included in this main non-physics comparison.}",
        r"\label{tab:selected_downstream}",
        r"\centering\scriptsize\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Task & Metric & \multicolumn{2}{c}{$k=2$} & \multicolumn{2}{c}{$k=3$} & \multicolumn{2}{c}{$k=4$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        f" & & {proposed} & Gain & {proposed} & Gain & {proposed} & Gain " + "\\\\",
        r"\midrule",
    ]
    names = {
        "strength_N100000": r"$10^5$-cycle strength",
        "life_at_mid_stress": "Life at service stress",
        "mean_log_life": "Mean curve life reserve",
        "mission_balanced_log_damage": "Mission-spectrum damage",
    }
    units = {
        "strength_N100000": r"MAE ($\log_{10}$ MPa)",
        "life_at_mid_stress": r"MAE ($\log_{10}$ cycles)",
        "mean_log_life": r"Integral MAE",
        "mission_balanced_log_damage": r"MAE ($\log_{10}D$)",
    }
    for task, _, _ in TASKS:
        cells = []
        for k in (2, 3, 4):
            pa = lookup[("test", k, task, proposed)]["mae"]
            base = min(lookup[("test", k, task, m)]["mae"] for m in BASELINES)
            gain = 100 * (base - pa) / base
            cells.extend([f"{pa:.3f}", f"{gain:.1f}\\%"])
        lines.append(f"{names[task]} & {units[task]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (args.output / "table_selected_downstream.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name", default="PA-CBB")
    main(parser.parse_args())
