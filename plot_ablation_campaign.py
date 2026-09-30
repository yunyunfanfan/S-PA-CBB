#!/usr/bin/env python3
"""Turn the completed multi-seed and sampling campaigns into paper artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#5B82B6"
LIGHT_BLUE = "#A9C4E4"
ROSE = "#CC7A7A"
PURPLE = "#8B82B8"
GOLD = "#E8B654"
DARK = "#243447"
GRID = "#D9DEE7"
COLORS = {
    "full": BLUE, "early_concat": ROSE, "no_physics": PURPLE,
    "no_shape": GOLD, "basquin_only": "#9AA3B2",
}
NAMES = {
    "full": "PA-CBB", "early_concat": "Early concat.",
    "no_physics": "No physics", "no_shape": "No shape loss",
    "basquin_only": "Basquin only",
}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3, "axes.titlesize": 9,
        "axes.labelsize": 8.4, "legend.fontsize": 7.2, "axes.linewidth": .8,
        "xtick.direction": "in", "ytick.direction": "in", "savefig.bbox": "tight",
    })


def panel_label(ax, value):
    ax.text(-.14, 1.06, value, transform=ax.transAxes, fontsize=11, fontweight="bold")


def save(fig, stem):
    fig.tight_layout(rect=(0, 0, 1, .965))
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400)):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi)
    plt.close(fig)


def seed_lookup(payload):
    return {(row["variant"], int(row["known"])): row for row in payload["seed_summary"]}


def architecture_figure(payload, output):
    lookup = seed_lookup(payload)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8))
    specs = [
        (("full", "early_concat"), "grid_mae_logN", "Grid MAE (log cycles)", "Control injection"),
        (("full", "early_concat"), "energy_score", "Energy Score", "Joint distribution"),
        (("full", "no_physics", "basquin_only"), "mae_logN", "Hidden-point MAE", "Physical state"),
        (("full", "no_physics", "basquin_only"), "grid_mae_logN", "Grid MAE", "Physical state"),
        (("full", "no_shape"), "slope_error", "Slope error", "Shape loss"),
        (("full", "no_shape"), "monotonic_violation", "Monotonic violation rate", "Shape loss"),
    ]
    x = np.arange(3)
    for index, (ax, (variants, metric, ylabel, title)) in enumerate(zip(axes.flat, specs)):
        width = .72 / len(variants)
        for j, variant in enumerate(variants):
            means, errors = [], []
            for known in (2, 3, 4):
                row = lookup[(variant, known)]
                means.append(float(row[metric]))
                errors.append(float(row.get(f"{metric}_seed_sd", 0.0)))
            ax.bar(x + (j - (len(variants) - 1) / 2) * width, means, width,
                   yerr=errors, capsize=2.2, color=COLORS[variant], edgecolor="white",
                   linewidth=.5, label=NAMES[variant])
        ax.set_xticks(x, [r"$k=2$", r"$k=3$", r"$k=4$"])
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=.55, alpha=.8); panel_label(ax, f"({chr(97+index)})")
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Matched ablations of conditional control, physics anchoring and curve-shape loss",
                 fontsize=11, fontweight="bold")
    save(fig, output)


def sampling_figure(payload, output):
    rows = payload["summary"]
    steps = [row for row in rows if row["sweep"] == "steps"]
    ensembles = [row for row in rows if row["sweep"] == "ensemble"]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.7))
    specs = [
        (steps, "mae_logN", "Hidden-point MAE", "Reverse steps"),
        (steps, "energy_score", "Energy Score", "Reverse steps"),
        (steps, "seconds_per_curve", "Inference time (s/curve)", "Reverse steps"),
        (ensembles, "crps_logN", "CRPS", "Ensemble size"),
        (ensembles, "coverage90", "Raw 90% coverage", "Ensemble size"),
        (ensembles, "seconds_per_curve", "Inference time (s/curve)", "Ensemble size"),
    ]
    for index, (ax, (selected, metric, ylabel, xlabel)) in enumerate(zip(axes.flat, specs)):
        x = [int(row["value"]) for row in selected]
        y = [float(row[metric]) for row in selected]
        ax.plot(x, y, marker="o", color=BLUE if index < 3 else PURPLE, lw=1.8, ms=4.5)
        ax.set_xticks(x); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if metric == "coverage90":
            ax.axhline(.9, color=DARK, ls="--", lw=1, label="nominal 0.90")
            ax.legend(frameon=False)
        ax.grid(True, color=GRID, lw=.55, alpha=.8); panel_label(ax, f"({chr(97+index)})")
    fig.suptitle("Reverse-step and posterior-ensemble sensitivity",
                 fontsize=11, fontweight="bold")
    save(fig, output)


def fmt(row, metric):
    value = float(row[metric])
    if row.get("seeds", 1) > 1 and f"{metric}_seed_sd" in row:
        return f"{value:.3f} $\\pm$ {float(row[f'{metric}_seed_sd']):.3f}"
    return f"{value:.3f}"


def ablation_table(payload, output):
    rows = payload["seed_summary"]
    order = {"full": 0, "early_concat": 1, "no_physics": 2, "no_shape": 3, "basquin_only": 4}
    rows = sorted(rows, key=lambda r: (order[r["variant"]], int(r["known"])))
    lines = [
        r"\begin{table*}[t]", r"\caption{Matched ablations on the fixed source-paper split under three reverse steps, 32 draws and $\eta=0.5$. Learned variants report mean $\pm$ between-seed standard deviation; Basquin is deterministic for a fixed mask.}",
        r"\label{tab:completed_ablation}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Variant & $k$ & Seeds & Point MAE & Grid MAE & Slope error & Violation rate & Energy Score \\", r"\midrule",
    ]
    previous = None
    for row in rows:
        variant = row["variant"]
        if previous is not None and variant != previous:
            lines.append(r"\addlinespace[2pt]")
        energy = fmt(row, "energy_score") if "energy_score" in row else "--"
        lines.append(f'{NAMES[variant]} & {int(row["known"])} & {int(row["seeds"])} & '
                     f'{fmt(row, "mae_logN")} & {fmt(row, "grid_mae_logN")} & '
                     f'{fmt(row, "slope_error")} & {fmt(row, "monotonic_violation")} & {energy} \\\\')
        previous = variant
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n")


def sampling_table(payload, output):
    lines = [
        r"\begin{table*}[t]", r"\caption{Reverse-step and ensemble-size sensitivity. Coverage is the raw, uncalibrated 90\% interval coverage.}",
        r"\label{tab:sampling_sensitivity}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Sweep & Value & Point MAE & CRPS & Energy Score & 90\% coverage & Width & Time (s/curve) \\", r"\midrule",
    ]
    for i, row in enumerate(payload["summary"]):
        if i == 5:
            lines.append(r"\addlinespace[3pt]")
        lines.append(f'{row["sweep"].title()} & {int(row["value"])} & {row["mae_logN"]:.3f} & '
                     f'{row["crps_logN"]:.3f} & {row["energy_score"]:.3f} & {row["coverage90"]:.3f} & '
                     f'{row["interval_width_logN"]:.3f} & {row["seconds_per_curve"]:.4f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n")


def bootstrap_table(payload, output):
    keep = {"grid_mae_logN": "Grid MAE", "slope_error": "Slope error",
            "energy_score": "Energy Score"}
    rows = [row for row in payload["paired_curve_bootstrap"] if row["metric"] in keep]
    names = {"no_physics-minus-full": "No physics $-$ PA-CBB",
             "early_concat-minus-full": "Early concat. $-$ PA-CBB",
             "no_shape-minus-full": "No shape loss $-$ PA-CBB"}
    lines = [
        r"\begin{table*}[t]", r"\caption{Paired curve-bootstrap differences relative to the complete model. Positive values favour PA-CBB for all listed loss metrics; intervals are 95\% confidence intervals from 10,000 curve resamples.}",
        r"\label{tab:ablation_bootstrap}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lclcc}", r"\toprule",
        r"Comparison & $k$ & Metric & Mean difference & 95\% CI \\", r"\midrule",
    ]
    previous = None
    for row in rows:
        comparison = row["comparison"]
        if previous is not None and comparison != previous:
            lines.append(r"\addlinespace[2pt]")
        lines.append(f'{names[comparison]} & {int(row["known"])} & {keep[row["metric"]]} & '
                     f'{row["mean_delta_other_minus_full"]:.4f} & '
                     f'[{row["ci_low"]:.4f}, {row["ci_high"]:.4f}] \\\\')
        previous = comparison
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n")


def main(args):
    setup(); args.output_dir.mkdir(parents=True, exist_ok=True)
    ablation = json.loads(args.ablation.read_text())
    sampling = json.loads(args.sampling.read_text())
    architecture_figure(ablation, args.output_dir / "completed_ablation")
    sampling_figure(sampling, args.output_dir / "sampling_sensitivity")
    ablation_table(ablation, args.output_dir / "table_completed_ablation.tex")
    bootstrap_table(ablation, args.output_dir / "table_ablation_bootstrap.tex")
    sampling_table(sampling, args.output_dir / "table_sampling_sensitivity.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
