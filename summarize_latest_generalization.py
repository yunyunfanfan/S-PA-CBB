#!/usr/bin/env python3
"""Summarize the mask-aligned latest S-PA-CBB pooled and LODO evaluations.

Model gates and residual weights are selected on validation data only.  Ridge and
MLP are deliberately excluded from the paper-facing comparator pool.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
POOL = ROOT / "final_protocol" / "selective_aligned_latest"
LODO = ROOT / "final_protocol" / "lodo_selective_aligned_latest"
METHODS = ["kNN", "RandomForest", "ExtraTrees", "Linear", "Polynomial", "PCHIP"]
KS = ["2", "3", "4"]
DOMAIN_ORDER = ["AM2022", "CMA2022", "Weld2025"]
LODO_TARGETS = [
    ("am2022_curves", "AM2022"),
    ("cma2022_curves", "CMA2022"),
    ("weld2025_disjoint_curves", "Weld2025"),
]

BLUE = "#3B6BA5"
LIGHT_BLUE = "#8FB3D9"
RED = "#C96F70"
GOLD = "#E7B44A"
PURPLE = "#8273B3"
GREY = "#777B84"


def load(path: Path):
    return json.loads(path.read_text())


def best_baseline(block: dict):
    scores = {m: block[m]["point"]["mae_logN"] for m in METHODS}
    method = min(scores, key=scores.get)
    return method, scores[method]


def collect_pool():
    selective = load(POOL / "selective" / "selective_main_metrics.json")
    baselines = load(POOL / "nonphysics_baselines_mask0.json")
    rows = []
    for domain in DOMAIN_ORDER:
        for k in KS:
            sm = selective["domain_summary"][domain][k]
            bm, bv = best_baseline(baselines["external"][domain]["known_points"][k])
            rows.append({
                "evaluation": "pooled-three-database",
                "database": domain,
                "eligible_curves": sm["curves"],
                "observed_points": int(k),
                "selected_gate": selective["gates"][k]["score"],
                "selected_alpha": selective["gates"][k]["alpha"],
                "activation_rate": sm["activation_rate"],
                "spacbb_point_mae": sm["point"]["mae_logN"],
                "best_nonphysics": bm,
                "best_nonphysics_point_mae": bv,
                "point_mae_improvement_pct": 100.0 * (bv - sm["point"]["mae_logN"]) / bv,
                "spacbb_grid_mae": sm["grid_mae_logN"],
                "raw_pacbb_grid_mae": sm["raw_pacbb_grid_mae_logN"],
                "basquin_grid_mae": sm["basquin_grid_mae_logN"],
                "raw_to_selective_grid_improvement_pct": 100.0 * (
                    sm["raw_pacbb_grid_mae_logN"] - sm["grid_mae_logN"]
                ) / sm["raw_pacbb_grid_mae_logN"],
            })
    return rows


def collect_lodo():
    rows = []
    for folder, label in LODO_TARGETS:
        selective = load(LODO / folder / "selective" / "selective_main_metrics.json")
        baselines = load(LODO / folder / "nonphysics_baselines_mask0.json")
        bdomain = baselines["external"][folder]
        for k in KS:
            sm = selective["summary"][k]
            bm, bv = best_baseline(bdomain["known_points"][k])
            rows.append({
                "evaluation": "leave-one-database-out",
                "database": label,
                "eligible_curves": sm["curves"],
                "observed_points": int(k),
                "selected_gate": selective["gates"][k]["score"],
                "selected_alpha": selective["gates"][k]["alpha"],
                "activation_rate": sm["activation_rate"],
                "spacbb_point_mae": sm["point"]["mae_logN"],
                "best_nonphysics": bm,
                "best_nonphysics_point_mae": bv,
                "point_mae_improvement_pct": 100.0 * (bv - sm["point"]["mae_logN"]) / bv,
                "spacbb_grid_mae": sm["grid_mae_logN"],
                "raw_pacbb_grid_mae": sm["raw_pacbb_grid_mae_logN"],
                "basquin_grid_mae": sm["basquin_grid_mae_logN"],
                "raw_to_selective_grid_improvement_pct": 100.0 * (
                    sm["raw_pacbb_grid_mae_logN"] - sm["grid_mae_logN"]
                ) / sm["raw_pacbb_grid_mae_logN"],
            })
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.dpi": 400,
    })


def heatmap(ax, matrix, row_labels, col_labels, title):
    # Fix the scientific comparison scale so a two-curve audit outlier does not
    # wash out practically relevant 5--45% improvements in the populated sets.
    vmax = 50.0
    image = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    color="white" if abs(val) > 0.55 * vmax else "#222222", fontsize=7.5)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Observed fatigue tests")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_pool(rows):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), gridspec_kw={"width_ratios": [1.15, 1.35, 1.35]})
    lookup = {(r["database"], r["observed_points"]): r for r in rows}
    matrix = np.array([[lookup[(d, k)]["point_mae_improvement_pct"] for k in (2, 3, 4)] for d in DOMAIN_ORDER])
    image = heatmap(axes[0], matrix, DOMAIN_ORDER, ["k=2", "k=3", "k=4"], "(a) Hidden-point MAE gain over best baseline")
    cb = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    cb.set_label("Relative improvement")

    x = np.arange(len(DOMAIN_ORDER))
    width = 0.22
    point_ceiling = max(lookup[(d, k)]["spacbb_point_mae"]
                        for d in DOMAIN_ORDER for k in (2, 3, 4))
    for idx, k in enumerate((2, 3, 4)):
        vals = [lookup[(d, k)]["spacbb_point_mae"] for d in DOMAIN_ORDER]
        axes[1].bar(x + (idx - 1) * width, vals, width, label=f"k={k}", color=[LIGHT_BLUE, PURPLE, BLUE][idx])
    axes[1].set_xticks(x, ["AM", "CMA", "Weld"])
    axes[1].set_ylabel(r"Hidden-point MAE ($\log_{10}N$)")
    axes[1].set_title("(b) Three-database test errors", loc="left", fontweight="bold")
    # Reserve a clean header band for the legend so it never covers the bars.
    axes[1].set_ylim(0, point_ceiling * 1.26)
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    axes[1].grid(axis="y", color="#D9DDE4", lw=0.55, alpha=0.8)
    axes[1].set_axisbelow(True)

    k = 4
    raw = [lookup[(d, k)]["raw_pacbb_grid_mae"] for d in DOMAIN_ORDER]
    sel = [lookup[(d, k)]["spacbb_grid_mae"] for d in DOMAIN_ORDER]
    basq = [lookup[(d, k)]["basquin_grid_mae"] for d in DOMAIN_ORDER]
    axes[2].bar(x - width, raw, width, label="Raw PA-CBB", color=RED, alpha=0.82)
    axes[2].bar(x, sel, width, label="Selective S-PA-CBB", color=BLUE)
    axes[2].bar(x + width, basq, width, label="Basquin state", color=GREY, alpha=0.72)
    axes[2].set_xticks(x, ["AM", "CMA", "Weld"])
    axes[2].set_ylabel(r"Complete-grid MAE ($\log_{10}N$)")
    axes[2].set_title("(c) Complete-curve reconstruction at k=4", loc="left", fontweight="bold")
    axes[2].set_ylim(0, max(raw + sel + basq) * 1.28)
    axes[2].legend(fontsize=7.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   columnspacing=0.9, handlelength=1.8)
    axes[2].grid(axis="y", color="#D9DDE4", lw=0.55, alpha=0.8)
    axes[2].set_axisbelow(True)
    fig.suptitle("Mask-aligned evaluation on three public fatigue databases", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(POOL / f"three_database_latest_comparison.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_lodo(rows):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    lookup = {(r["database"], r["observed_points"]): r for r in rows}
    targets = [label for _, label in LODO_TARGETS]
    x = np.arange(len(targets))
    colors = [LIGHT_BLUE, PURPLE, BLUE]
    width = 0.23

    for idx, k in enumerate((2, 3, 4)):
        ours = [lookup[(d, k)]["spacbb_point_mae"] for d in targets]
        best = [lookup[(d, k)]["best_nonphysics_point_mae"] for d in targets]
        offset = (idx - 1) * width
        axes[0].bar(x + offset, ours, width, color=colors[idx], label=f"S-PA-CBB, k={k}")
        axes[0].scatter(x + offset, best, s=35, marker="D", facecolors="white", edgecolors="#252525", linewidths=0.9, zorder=3)
    axes[0].set_xticks(x, targets)
    axes[0].set_ylabel(r"Hidden-point MAE ($\log_{10}N$)")
    axes[0].set_title("(a) Unseen-database hidden-point prediction", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=1)
    axes[0].text(0.98, 0.97, "◇ best non-physics baseline", transform=axes[0].transAxes, ha="right", va="top", fontsize=7)

    matrix = np.array([[lookup[(d, k)]["point_mae_improvement_pct"] for k in (2, 3, 4)] for d in targets])
    image = heatmap(axes[1], matrix, targets, ["k=2", "k=3", "k=4"], "(b) Relative gain in the unseen database")
    cb = fig.colorbar(image, ax=axes[1], fraction=0.045, pad=0.03)
    cb.set_label("Relative improvement")

    k = 4
    raw = [lookup[(d, k)]["raw_pacbb_grid_mae"] for d in targets]
    sel = [lookup[(d, k)]["spacbb_grid_mae"] for d in targets]
    basq = [lookup[(d, k)]["basquin_grid_mae"] for d in targets]
    axes[2].bar(x - width, raw, width, label="Raw PA-CBB", color=RED, alpha=0.82)
    axes[2].bar(x, sel, width, label="Selective S-PA-CBB", color=BLUE)
    axes[2].bar(x + width, basq, width, label="Basquin state", color=GREY, alpha=0.72)
    axes[2].set_xticks(x, targets)
    axes[2].set_ylabel(r"Complete-grid MAE ($\log_{10}N$)")
    axes[2].set_title("(c) Unseen-database curve reconstruction at k=4", loc="left", fontweight="bold")
    axes[2].legend(fontsize=7.4)
    for ax in (axes[0], axes[2]):
        ax.grid(axis="y", color="#D9DDE4", lw=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("True leave-one-database-out generalization", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(LODO / f"lodo_latest_comparison.{suffix}", bbox_inches="tight")
    plt.close(fig)


def write_tex(pool_rows, lodo_rows):
    def section(rows, caption, label):
        lines = [
            r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.2pt}",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r"Database & $k$ & Curves & S-PA-CBB & Best non-physics & Gain (\%) & Grid MAE & Raw PA-CBB \\",
            r"\midrule",
        ]
        for row in rows:
            db = row["database"].replace("NIMS-derived", "NIMS$^{*}$").replace("HEA2022", "HEA$^{*}$")
            base = row["best_nonphysics"].replace("RandomForest", "RF").replace("ExtraTrees", "ET").replace("Polynomial", "Poly.")
            lines.append(
                f"{db} & {row['observed_points']} & {row['eligible_curves']} & "
                f"{row['spacbb_point_mae']:.3f} & {base}: {row['best_nonphysics_point_mae']:.3f} & "
                f"{row['point_mae_improvement_pct']:+.1f} & {row['spacbb_grid_mae']:.3f} & "
                f"{row['raw_pacbb_grid_mae']:.3f} \\\\" 
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        if rows[0]["evaluation"] == "leave-one-database-out":
            lines.append(r"\begin{minipage}{0.98\textwidth}\footnotesize Best non-physics excludes Ridge and MLP. Each target database is absent from model fitting and gate selection.\end{minipage}")
        else:
            lines.append(r"\begin{minipage}{0.98\textwidth}\footnotesize Best non-physics excludes Ridge and MLP. Gates and correction weights are selected exclusively on validation curves.\end{minipage}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    (POOL / "three_database_latest_table.tex").write_text(section(
        pool_rows,
        "Mask-aligned three-database evaluation. Point MAE is evaluated only at hidden measurements; grid MAE evaluates complete-curve reconstruction.",
        "tab:three_database_latest",
    ))
    (LODO / "lodo_latest_table.tex").write_text(section(
        lodo_rows,
        "True leave-one-database-out evaluation under identical sparse masks. The best non-physics comparator is selected separately within the prespecified baseline set.",
        "tab:lodo_latest",
    ))


def write_report(pool_rows, lodo_rows):
    def line(row):
        return (
            f"| {row['database']} | {row['observed_points']} | {row['eligible_curves']} | "
            f"{row['spacbb_point_mae']:.3f} | {row['best_nonphysics']} ({row['best_nonphysics_point_mae']:.3f}) | "
            f"{row['point_mae_improvement_pct']:+.1f}% | {row['spacbb_grid_mae']:.3f} | "
            f"{row['raw_pacbb_grid_mae']:.3f} | {row['basquin_grid_mae']:.3f} |"
        )
    header = "| Database | k | Curves | S-PA-CBB point MAE | Best non-physics | Gain | Selective grid | Raw grid | Basquin grid |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    pool_good = sum(r["point_mae_improvement_pct"] > 0 for r in pool_rows)
    lodo_good = sum(r["point_mae_improvement_pct"] > 0 for r in lodo_rows)
    report = f"""# Latest mask-aligned three-database and LODO evaluation

All gates, thresholds, and residual weights were selected using validation curves only. The test masks are identical across methods. Ridge and MLP are excluded from the paper-facing comparator pool as requested.

## Pooled three-database evaluation

{header}
{chr(10).join(line(r) for r in pool_rows)}

S-PA-CBB wins {pool_good}/{len(pool_rows)} database-budget cells in hidden-point MAE.

## True leave-one-database-out evaluation

{header}
{chr(10).join(line(r) for r in lodo_rows)}

S-PA-CBB wins {lodo_good}/{len(lodo_rows)} unseen-database budget cells in hidden-point MAE. The single loss is CMA2022 at k=2. Selective correction reduces raw PA-CBB grid MAE in every LODO cell, but it does not uniformly improve over the Basquin state on the complete grid. The defensible claim is therefore improved sparse hidden-point prediction and stabilization of the raw generator, not universal replacement of the physics baseline under domain shift.

## Selection protocol

- Reverse steps: 3; posterior samples: 32; eta: 0.5.
- Sparse-mask seed: 20261030; posterior-draw seed: 20261110.
- One validation-selected gate and correction weight per observation budget.
- Test data were opened once after model and gate selection; no test-set cherry-picking was used.
"""
    (POOL / "THREE_DATABASE_AND_LODO_REPORT.md").write_text(report)


def main():
    pool_rows = collect_pool()
    lodo_rows = collect_lodo()
    write_csv(POOL / "three_database_summary.csv", pool_rows)
    write_csv(LODO / "lodo_summary.csv", lodo_rows)
    plot_pool(pool_rows)
    plot_lodo(lodo_rows)
    write_tex(pool_rows, lodo_rows)
    write_report(pool_rows, lodo_rows)
    print(f"wrote {len(pool_rows)} pooled rows and {len(lodo_rows)} LODO rows")


if __name__ == "__main__":
    main()
