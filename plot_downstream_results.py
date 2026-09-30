#!/usr/bin/env python3
"""Publication figures and LaTeX tables for the two downstream experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#5B82B6"; LIGHT_BLUE = "#A9C4E4"; ROSE = "#CC7A7A"
PURPLE = "#8B82B8"; GOLD = "#E8B654"; DARK = "#243447"; GRID = "#D9DEE7"
KCOLORS = {2: LIGHT_BLUE, 3: PURPLE, 4: ROSE}
POLICY_COLORS = {
    "random": "#D8DCE5", "largest_gap": "#A9C4E4", "space_filling": "#8FB5D8",
    "raw_variance": "#8B82B8", "conformal_width": "#D9A0A0", "eivr": BLUE,
}
POLICY_LABELS = {
    "random": "Rand.", "largest_gap": "Gap", "space_filling": "Space",
    "raw_variance": "Var.", "conformal_width": "Conf.", "eivr": "EIVR",
}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "savefig.bbox": "tight",
    })


def label(ax, text):
    ax.text(-0.13, 1.06, text, transform=ax.transAxes, fontsize=11, fontweight="bold")


def strength_figure(rows, output):
    targets = [100000, 1000000, 10000000]
    specs = [
        ("stress_mae_mpa", "Stress MAE (MPa)", None),
        ("relative_stress_error", "Relative stress error", None),
        ("spearman_rho", "Spearman $\\rho$", None),
        ("coverage90", "Empirical 90% coverage", 0.9),
        ("interval_width_mpa", "Interval width (MPa)", None),
        ("pairwise_ranking_accuracy", "Pairwise ranking accuracy", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.7))
    for panel, (ax, (metric, ylabel, reference)) in enumerate(zip(axes.flat, specs)):
        for known in (2, 3, 4):
            values = [next(row[metric] for row in rows
                           if row["known"] == known and row["target_cycles"] == target)
                      for target in targets]
            ax.plot(range(3), values, marker="o", lw=1.8, ms=4.5,
                    color=KCOLORS[known], label=f"$k={known}$")
        if reference is not None:
            ax.axhline(reference, color=DARK, ls="--", lw=1, label="nominal 0.90")
        ax.set_xticks(range(3), [r"$10^5$", r"$10^6$", r"$10^7$"])
        ax.set_xlabel("Target life (cycles)"); ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, lw=0.55, alpha=0.8); label(ax, f"({chr(97 + panel)})")
        if panel == 0:
            ax.legend(frameon=False, ncol=3, loc="best")
        if metric in ("spearman_rho", "coverage90", "pairwise_ranking_accuracy"):
            ax.set_ylim(0.82, 1.005)
    fig.suptitle("Fatigue-strength retrieval from sparse conditional curve ensembles",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400)):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=dpi)
    plt.close(fig)


def active_figure(summary, cases, output):
    policies = [row["policy"] for row in summary]
    labels = [POLICY_LABELS[p] for p in policies]
    colors = [POLICY_COLORS[p] for p in policies]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.9))
    bar_specs = [
        ("grid_mae_budget_auc", "Grid-MAE--budget AUC", "lower is better"),
        ("slope_error_after_two_tests", "Slope error after two tests", "lower is better"),
        ("strength_error_mpa_N1e6_after_two_tests", r"$10^6$-cycle strength error (MPa)", "lower is better"),
        ("mean_tests_to_threshold", "Tests to MAE $\\leq0.25$", "lower is better"),
        ("threshold_success_rate", "Threshold success rate", "higher is better"),
    ]
    ax = axes[0, 0]
    for policy in policies:
        means = [np.mean([float(row["grid_mae_logN"]) for row in cases
                          if row["policy"] == policy and int(row["budget"]) == budget])
                 for budget in (0, 1, 2)]
        ax.plot((0, 1, 2), means, marker="o", lw=1.5, color=POLICY_COLORS[policy],
                label=policy.replace("_", " "))
    ax.set_xticks((0, 1, 2)); ax.set_xlabel("Additional fatigue tests")
    ax.set_ylabel("Grid MAE (log cycles)"); ax.grid(True, color=GRID, lw=.55)
    ax.legend(frameon=False, fontsize=6.4, ncol=2); label(ax, "(a)")
    for panel, (ax, (metric, ylabel, note)) in enumerate(zip(axes.flat[1:], bar_specs), start=1):
        values = [float(next(row[metric] for row in summary if row["policy"] == policy)) for policy in policies]
        ax.bar(range(len(policies)), values, color=colors, edgecolor="white", linewidth=.6)
        ax.set_xticks(range(len(policies)), labels, rotation=0, ha="center")
        ax.set_ylabel(ylabel); ax.grid(axis="y", color=GRID, lw=.55)
        ax.text(.98, .96, note, transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color="#667085")
        label(ax, f"({chr(97 + panel)})")
    fig.suptitle("Uncertainty-guided selection of the next fatigue experiment",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400)):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=dpi)
    plt.close(fig)


def strength_table(rows, output):
    lines = [
        r"\begin{table*}[t]", r"\caption{Fatigue-strength retrieval from calibrated PA-CBB ensembles. Relative error, coverage and ranking accuracy are fractions.}",
        r"\label{tab:downstream_strength}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{cccccccc}", r"\toprule",
        r"$k$ & Target life & Cases & MAE (MPa) & Relative error & Spearman $\rho$ & 90\% coverage / width (MPa) & Pairwise accuracy \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f'{row["known"]} & $10^{{{int(np.log10(row["target_cycles"]))}}}$ & {row["cases"]} & '
                     f'{row["stress_mae_mpa"]:.2f} & {row["relative_stress_error"]:.3f} & '
                     f'{row["spearman_rho"]:.3f} & {row["coverage90"]:.3f} / {row["interval_width_mpa"]:.1f} & '
                     f'{row["pairwise_ranking_accuracy"]:.3f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n")


def strength_baseline_table(model_rows, baseline_rows, output):
    """Nine-condition point-estimation comparison; physics-only Basquin stays in ablations."""
    methods = ["PA-CBB", "Linear", "PCHIP", "Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP"]
    lookup = {("PA-CBB", int(r["known"]), int(r["target_cycles"])): r for r in model_rows}
    lookup.update({(r["method"], int(r["known"]), int(r["target_cycles"])): r for r in baseline_rows
                   if r["method"] in methods})
    conditions = [(k, n) for k in (2, 3, 4) for n in (100000, 1000000, 10000000)]
    best = {c: min(float(lookup[(m, *c)]["stress_mae_mpa"]) for m in methods) for c in conditions}
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Stress MAE (MPa) for fatigue-strength retrieval. Each column uses identical held-out curves and sparse masks. Basquin is reserved for the physics ablation. Best values are bold.}",
        r"\label{tab:downstream_strength_baselines}", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lccc|ccc|ccc}", r"\toprule",
        r"& \multicolumn{3}{c}{$k=2$} & \multicolumn{3}{c}{$k=3$} & \multicolumn{3}{c}{$k=4$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Method & $10^5$ & $10^6$ & $10^7$ & $10^5$ & $10^6$ & $10^7$ & $10^5$ & $10^6$ & $10^7$ \\",
        r"\midrule",
    ]
    display = {"RandomForest": "Random forest", "ExtraTrees": "Extra Trees", "kNN": "$k$NN"}
    for method in methods:
        vals = []
        for condition in conditions:
            value = float(lookup[(method, *condition)]["stress_mae_mpa"])
            cell = f"{value:.2f}"
            if np.isclose(value, best[condition]):
                cell = rf"\textbf{{{cell}}}"
            vals.append(cell)
        lines.append(f"{display.get(method, method)} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n")


def active_table(rows, output):
    lines = [
        r"\begin{table*}[t]", r"\caption{Offline active-testing comparison from the same two initial anchors. AUC is normalized over two additional tests; the threshold is grid MAE $\leq0.25$.}",
        r"\label{tab:downstream_active}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Policy & MAE AUC & MAE after 1 & MAE after 2 & Slope error & Strength error (MPa) & Tests to threshold & Success rate \\",
        r"\midrule",
    ]
    for row in rows:
        name = row["policy"].replace("_", " ").title().replace("Eivr", "EIVR")
        lines.append(f'{name} & {row["grid_mae_budget_auc"]:.3f} & {row["mae_after_one_test"]:.3f} & '
                     f'{row["mae_after_two_tests"]:.3f} & {row["slope_error_after_two_tests"]:.3f} & '
                     f'{row["strength_error_mpa_N1e6_after_two_tests"]:.2f} & '
                     f'{row["mean_tests_to_threshold"]:.2f} & {row["threshold_success_rate"]:.3f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n")


def main(args):
    setup(); args.output_dir.mkdir(parents=True, exist_ok=True)
    strength = json.loads(args.strength.read_text())["summary"]
    active = json.loads(args.active.read_text())["summary"]
    baseline_rows = json.loads(args.strength_baselines.read_text())["summary"] if args.strength_baselines else None
    with args.active_cases.open() as handle:
        cases = list(csv.DictReader(handle))
    strength_figure(strength, args.output_dir / "downstream_fatigue_strength")
    active_figure(active, cases, args.output_dir / "downstream_active_selection")
    strength_table(strength, args.output_dir / "table_downstream_strength.tex")
    if baseline_rows is not None:
        strength_baseline_table(strength, baseline_rows,
                                args.output_dir / "table_downstream_strength_baselines.tex")
    active_table(active, args.output_dir / "table_downstream_active.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=Path, required=True)
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--active-cases", type=Path, required=True)
    parser.add_argument("--strength-baselines", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
