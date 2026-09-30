#!/usr/bin/env python3
"""Aggregate and plot the final shrinkage, monotonicity, mask and ensemble audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE, LIGHT_BLUE, ROSE, PURPLE, GOLD = "#5B82B6", "#A9C4E4", "#CC7A7A", "#8B82B8", "#E8B654"
GRID, DARK = "#D9DEE7", "#243447"
COMMON = {20261020, 20261021, 20261022}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.titlesize": 9, "axes.labelsize": 8.4,
                         "legend.fontsize": 7.1, "axes.linewidth": .8,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "savefig.bbox": "tight"})


def save(fig, stem):
    fig.tight_layout()
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400)):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi)
    plt.close(fig)


def load_sensitivity(root):
    rows = []
    for path in sorted(root.glob("full_seed*/metrics.json")):
        payload = json.loads(path.read_text()); seed = int(payload["seed"])
        rows.extend([{**row, "seed": seed} for row in payload["summary"]])
    return rows


def aggregate(rows, axis, common_only=False):
    selected = [r for r in rows if r["axis"] == axis and (not common_only or r["seed"] in COMMON)]
    output = []
    for variant, known in sorted({(r["variant"], int(r["known"])) for r in selected}):
        group = [r for r in selected if r["variant"] == variant and int(r["known"]) == known]
        item = {"variant": variant, "known": known, "seeds": len(group)}
        for metric in ("mae_logN", "interpolation_mae_logN", "extrapolation_mae_logN",
                       "grid_mae_logN", "slope_error", "monotonic_violation",
                       "energy_score", "coverage90", "anchor_shift_logN"):
            values = [r[metric] for r in group if r.get(metric) is not None and np.isfinite(r[metric])]
            item[metric] = float(np.mean(values)) if values else None
            item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def lookup(rows):
    return {(r["variant"], r["known"]): r for r in rows}


def physics_monotonicity_figure(shrink, monotonic, output):
    shrink_l, mono_l = lookup(shrink), lookup(monotonic)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.5))
    shrink_order = ["no_shrink", "weak_shrink", "current_shrink", "strong_shrink"]
    shrink_labels = ["0", "0.001", "0.01\n(current)", "0.1"]
    colors = [LIGHT_BLUE, BLUE, PURPLE, ROSE]
    for ax, metric, title in zip(axes[0], ("grid_mae_logN", "slope_error", "energy_score"),
                                 ("Full-grid MAE", "Slope error", "Energy Score")):
        x = np.arange(4)
        for j, known in enumerate((2, 3, 4)):
            vals = [shrink_l[(v, known)][metric] for v in shrink_order]
            ax.plot(x, vals, marker="o", lw=1.6, color=(BLUE, PURPLE, ROSE)[j], label=f"k={known}")
        ax.set_xticks(x, shrink_labels); ax.set_xlabel("Slope ridge strength"); ax.set_ylabel(title)
        ax.grid(axis="y", color=GRID, lw=.55); ax.legend(frameon=False)

    mono_order = ["no_shape_loss", "soft_shape_loss", "hard_projection"]
    mono_labels = ["No shape\nloss", "Soft shape\nloss", "Hard\nprojection"]
    for ax, metric, title in zip(axes[1], ("grid_mae_logN", "monotonic_violation", "energy_score"),
                                 ("Full-grid MAE", "Violation rate", "Energy Score")):
        x = np.arange(3); width = .23
        for j, known in enumerate((2, 3, 4)):
            vals = [mono_l[(v, known)][metric] for v in mono_order]
            ax.bar(x + (j - 1) * width, vals, width, color=(BLUE, PURPLE, ROSE)[j],
                   edgecolor="white", label=f"k={known}")
        ax.set_xticks(x, mono_labels); ax.set_ylabel(title); ax.grid(axis="y", color=GRID, lw=.55)
        ax.legend(frameon=False)
    for index, ax in enumerate(axes.flat):
        ax.text(-.14, 1.04, f"({chr(97 + index)})", transform=ax.transAxes,
                fontsize=10.5, fontweight="bold")
    save(fig, output / "physics_monotonicity_sensitivity")


def mask_figure(mask_rows, output):
    names = ["random", "uniform_interior", "high_stress", "low_stress", "adjacent_center", "endpoints"]
    labels = ["Random", "Uniform", "High-stress", "Low-stress", "Adjacent", "Endpoints"]
    lk = lookup(mask_rows)
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 5.8))
    specs = [("mae_logN", "Hidden-point MAE"), ("interpolation_mae_logN", "Interpolation MAE"),
             ("extrapolation_mae_logN", "Extrapolation MAE"), ("energy_score", "Energy Score")]
    for index, (ax, (metric, title)) in enumerate(zip(axes.flat, specs)):
        matrix = np.asarray([[lk[(name, known)][metric] if lk[(name, known)][metric] is not None else np.nan
                              for known in (2, 3, 4)] for name in names])
        image = ax.imshow(matrix, cmap="RdPu", aspect="auto")
        ax.set_xticks(range(3), ["k=2", "k=3", "k=4"]); ax.set_yticks(range(6), labels)
        ax.set_title(title, fontweight="bold")
        for i in range(6):
            for j in range(3):
                text = "--" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.3f}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if np.isfinite(matrix[i, j]) and matrix[i, j] > np.nanmedian(matrix) else DARK,
                        fontsize=7.2)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.03)
        ax.text(-.12, 1.04, f"({chr(97 + index)})", transform=ax.transAxes,
                fontsize=10.5, fontweight="bold")
    save(fig, output / "mask_location_robustness")


def fmt(row, metric):
    value = row.get(metric)
    if value is None:
        return "--"
    return f"{value:.3f} $\\pm$ {row.get(metric + '_sd', 0.0):.3f}"


def write_sensitivity_tables(shrink, monotonic, masks, output):
    names = {"no_shrink": "No shrinkage", "weak_shrink": "Weak ($10^{-3}$)",
             "current_shrink": "Current ($10^{-2}$)", "strong_shrink": "Strong ($10^{-1}$)",
             "no_shape_loss": "No shape loss", "soft_shape_loss": "Soft shape loss",
             "hard_projection": "Hard monotone projection"}
    lines = [r"\begin{table*}[t]", r"\caption{Physical-slope shrinkage under the frozen three-step sampler. Values are mean $\pm$ standard deviation across five full-model seeds.}",
             r"\label{tab:shrinkage}", r"\centering\scriptsize\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{lccccc}", r"\toprule", r"Variant & $k$ & Hidden MAE $\downarrow$ & Grid MAE $\downarrow$ & Slope error $\downarrow$ & Energy Score $\downarrow$ \\", r"\midrule"]
    for row in shrink:
        lines.append(f'{names[row["variant"]]} & {row["known"]} & {fmt(row,"mae_logN")} & {fmt(row,"grid_mae_logN")} & {fmt(row,"slope_error")} & {fmt(row,"energy_score")} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "table_shrinkage.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{table*}[t]", r"\caption{Monotonicity treatments on three common paired seeds. Hard projection is weighted toward observed anchors; anchor shift reports its mean absolute change at those anchors.}",
             r"\label{tab:monotonicity}", r"\centering\scriptsize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule", r"Variant & $k$ & Grid MAE $\downarrow$ & Slope error $\downarrow$ & Violation rate $\downarrow$ & Energy Score $\downarrow$ & Anchor shift $\downarrow$ \\", r"\midrule"]
    for row in monotonic:
        lines.append(f'{names[row["variant"]]} & {row["known"]} & {fmt(row,"grid_mae_logN")} & {fmt(row,"slope_error")} & {fmt(row,"monotonic_violation")} & {fmt(row,"energy_score")} & {fmt(row,"anchor_shift_logN")} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "table_monotonicity.tex").write_text("\n".join(lines) + "\n")

    mask_names = {"random": "Random", "uniform_interior": "Uniform interior", "high_stress": "High-stress end",
                  "low_stress": "Low-stress end", "adjacent_center": "Adjacent center", "endpoints": "Endpoints"}
    lines = [r"\begin{table*}[t]", r"\caption{Sparse-anchor location robustness across five full-model seeds. Interpolation and extrapolation are defined relative to the observed stress span; endpoint masks have no extrapolation points.}",
             r"\label{tab:mask_robustness}", r"\centering\scriptsize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule", r"Mask pattern & $k$ & Hidden MAE $\downarrow$ & Interpolation MAE $\downarrow$ & Extrapolation MAE $\downarrow$ & Grid MAE $\downarrow$ & Energy Score $\downarrow$ \\", r"\midrule"]
    for row in masks:
        lines.append(f'{mask_names[row["variant"]]} & {row["known"]} & {fmt(row,"mae_logN")} & {fmt(row,"interpolation_mae_logN")} & {fmt(row,"extrapolation_mae_logN")} & {fmt(row,"grid_mae_logN")} & {fmt(row,"energy_score")} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "table_mask_robustness.tex").write_text("\n".join(lines) + "\n")


def downstream_outputs(strength_path, active_root, output):
    strength = json.loads(strength_path.read_text())["summary"]
    sizes = (1, 4, 8, 16, 32, 64); macro = []
    for size in sizes:
        rows = [r for r in strength if r["ensemble"] == size]
        item = {"ensemble": size}
        for metric in ("stress_mae_mpa", "raw_coverage90", "conformal_coverage90",
                       "raw_interval_width_mpa", "conformal_interval_width_mpa",
                       "pairwise_ranking_accuracy"):
            item[metric] = float(np.mean([r[metric] for r in rows]))
        active = json.loads((active_root / f"active_{size}" / "active_selection_summary.json").read_text())
        eivr = next(r for r in active["summary"] if r["policy"] == "eivr")
        for key, value in eivr.items():
            if key not in ("policy", "curves"):
                item["active_" + key] = value
        macro.append(item)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.4)); x = np.arange(len(sizes))
    specs = [("stress_mae_mpa", "Fatigue-strength MAE (MPa)"),
             ("coverage", "90% fatigue-strength coverage"),
             ("width", "Fatigue-strength interval width (MPa)"),
             ("active_grid_mae_budget_auc", "EIVR grid-MAE AUC"),
             ("active_mae_after_two_tests", "EIVR MAE after two tests"),
             ("active_threshold_success_rate", "EIVR threshold success rate")]
    for index, (ax, (metric, title)) in enumerate(zip(axes.flat, specs)):
        if metric == "coverage":
            ax.plot(x, [r["raw_coverage90"] for r in macro], "o-", color=ROSE, label="Raw")
            ax.plot(x, [r["conformal_coverage90"] for r in macro], "o-", color=BLUE, label="Conformal")
            ax.axhline(.9, color=DARK, ls="--", lw=1, label="Nominal")
            ax.legend(frameon=False)
        elif metric == "width":
            ax.plot(x, [r["raw_interval_width_mpa"] for r in macro], "o-", color=ROSE, label="Raw")
            ax.plot(x, [r["conformal_interval_width_mpa"] for r in macro], "o-", color=BLUE, label="Conformal")
            ax.legend(frameon=False)
        else:
            ax.plot(x, [r[metric] for r in macro], "o-", color=PURPLE if index >= 3 else BLUE, lw=1.7)
        ax.set_xticks(x, sizes); ax.set_xlabel("Generated curves"); ax.set_ylabel(title)
        ax.grid(True, color=GRID, lw=.55); ax.text(-.14, 1.04, f"({chr(97 + index)})",
                                                   transform=ax.transAxes, fontsize=10.5, fontweight="bold")
    save(fig, output / "downstream_ensemble_sensitivity")

    lines = [r"\begin{table*}[t]", r"\caption{Effect of posterior ensemble size on fatigue-strength retrieval and EIVR active testing. Strength metrics are macro-averaged over three anchor budgets and three target lives.}",
             r"\label{tab:downstream_ensemble}", r"\centering\scriptsize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{ccccccccc}", r"\toprule", r"Samples & Strength MAE & Raw cov. & Conf. cov. & Conf. width & EIVR AUC & MAE +2 & Slope +2 & Success \\", r"\midrule"]
    for r in macro:
        lines.append(f'{r["ensemble"]} & {r["stress_mae_mpa"]:.2f} & {r["raw_coverage90"]:.3f} & {r["conformal_coverage90"]:.3f} & {r["conformal_interval_width_mpa"]:.1f} & {r["active_grid_mae_budget_auc"]:.3f} & {r["active_mae_after_two_tests"]:.3f} & {r["active_slope_error_after_two_tests"]:.3f} & {r["active_threshold_success_rate"]:.3f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "table_downstream_ensemble.tex").write_text("\n".join(lines) + "\n")
    (output / "downstream_ensemble_macro.json").write_text(json.dumps(macro, indent=2))


def main(args):
    setup(); args.output.mkdir(parents=True, exist_ok=True)
    sensitivity = load_sensitivity(args.sensitivity)
    shrink = aggregate(sensitivity, "shrinkage")
    masks = aggregate(sensitivity, "mask_location")
    monotonic = aggregate(sensitivity, "monotonicity", common_only=True)
    ablation = json.loads(args.ablation.read_text())["seed_summary"]
    for row in ablation:
        if row["variant"] == "no_shape":
            monotonic.append({"variant": "no_shape_loss", "known": int(row["known"]),
                              "seeds": int(row["seeds"]), **{key: row.get(key) for key in
                              ("grid_mae_logN", "slope_error", "monotonic_violation", "energy_score")},
                              **{key + "_sd": row.get(key + "_seed_sd", 0.0) for key in
                              ("grid_mae_logN", "slope_error", "monotonic_violation", "energy_score")},
                              "anchor_shift_logN": 0.0, "anchor_shift_logN_sd": 0.0})
    monotonic.sort(key=lambda r: ({"no_shape_loss": 0, "soft_shape_loss": 1, "hard_projection": 2}[r["variant"]], r["known"]))
    physics_monotonicity_figure(shrink, monotonic, args.output)
    mask_figure(masks, args.output)
    write_sensitivity_tables(shrink, monotonic, masks, args.output)
    downstream_outputs(args.strength, args.active, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--strength", type=Path, required=True)
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
