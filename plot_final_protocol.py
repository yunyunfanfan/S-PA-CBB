#!/usr/bin/env python3
"""Publication figures and tables for the unified 3-step protocol."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final_protocol" / "publication"
BLUE, ROSE, PURPLE, GOLD, GREY = "#567BA8", "#D27D7D", "#8B80B8", "#E8B14D", "#8A94A5"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 400)):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def source_rows():
    ours = json.loads((ROOT / "final_protocol/source_3step_calibrated/metrics.json").read_text())["summary"]
    extra = json.loads((ROOT / "final_protocol/probability_physics/source_baselines.json").read_text())["known_points"]
    return ours, extra


def plot_physics_probability():
    ours, extra = source_rows(); ks = [2, 3, 4]
    physics = ["Naive Basquin", "Extreme-two Basquin", "Fixed-slope Wohler", "Fixed-limit Stromeyer"]
    probability = ["PA-CBB", "GP-RBF", "Bayesian Basquin", "ExtraTrees ensemble"]
    colors = [BLUE, GREY, PURPLE, ROSE]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2))
    x = np.arange(len(ks)); width = 0.18
    for index, name in enumerate(physics):
        values = [extra[str(k)]["weak_physics"][name]["point"]["mae_logN"] for k in ks]
        axes[0, 0].bar(x + (index - 1.5) * width, values, width, color=colors[index], label=name)
        values = [extra[str(k)]["weak_physics"][name]["grid_mae_logN"] for k in ks]
        axes[0, 1].bar(x + (index - 1.5) * width, values, width, color=colors[index])
        values = [extra[str(k)]["weak_physics"][name]["slope_error"] for k in ks]
        axes[0, 2].bar(x + (index - 1.5) * width, values, width, color=colors[index])
    for ax, title, ylabel in zip(axes[0], ("Hidden-point error", "Full-grid error", "Slope fidelity"),
                                  ("MAE (log cycles)", "Grid MAE", "Slope error")):
        ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel); ax.set_xticks(x, ["2", "3", "4"]); ax.set_xlabel("Observed anchors"); ax.grid(axis="y", alpha=.22)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    metrics = (("crps_logN", "CRPS"), ("energy_score", "Energy Score"), ("conformal_coverage90", "Conformal 90% coverage"))
    for panel, (metric, title) in enumerate(metrics):
        ax = axes[1, panel]
        for index, name in enumerate(probability):
            if name == "PA-CBB":
                values = [ours[str(k)][metric] for k in ks]
            else:
                values = [extra[str(k)]["probabilistic"][name][metric] for k in ks]
            ax.plot(ks, values, "o-", color=colors[index], label=name, lw=1.8)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Observed anchors"); ax.set_xticks(ks); ax.grid(alpha=.22)
        if "coverage" in metric: ax.axhline(.9, color="black", ls="--", lw=1); ax.set_ylim(.82, .95)
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Unified three-step protocol: weak physics and probabilistic baselines", fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, "physics_probability_baselines")


def write_tables():
    ours, extra = source_rows()
    lines = [r"\begin{table*}[t]", r"\caption{Probabilistic baselines under the unified three-step protocol. Coverage is reported after validation-only split-conformal calibration. Lower is better for MAE, CRPS, Energy Score and width.}", r"\label{tab:probabilistic_baselines}", r"\centering\scriptsize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccccccc}", r"\toprule", r"Method & $k$ & MAE $\downarrow$ & CRPS $\downarrow$ & Energy $\downarrow$ & Raw cov. & Conf. cov. & Conf. width $\downarrow$ \\", r"\midrule"]
    names = ["PA-CBB", "GP-RBF", "Bayesian Basquin", "ExtraTrees ensemble"]
    labels = {"PA-CBB": r"PA-CBB (ours)", "GP-RBF": "GP-RBF", "Bayesian Basquin": "Bayesian Basquin", "ExtraTrees ensemble": "ExtraTrees ensemble"}
    for k in (2, 3, 4):
        for name in names:
            row = ours[str(k)] if name == "PA-CBB" else extra[str(k)]["probabilistic"][name]
            point = row["point"]["mae_logN"]
            lines.append(f'{labels[name]} & {k} & {point:.3f} & {row["crps_logN"]:.3f} & {row["energy_score"]:.3f} & {row["raw_coverage90"]:.3f} & {row["conformal_coverage90"]:.3f} & {row["conformal_width90_logN"]:.3f} \\\\')
        if k != 4: lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_probabilistic_baselines.tex").write_text("\n".join(lines))

    lines = [r"\begin{table*}[t]", r"\caption{Weak physical baselines on the same held-out curves and sparse masks. These intentionally simple laws test whether any physics-shaped curve is sufficient.}", r"\label{tab:weak_physics}", r"\centering\scriptsize\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lccccc}", r"\toprule", r"Method & $k$ & Hidden MAE $\downarrow$ & Grid MAE $\downarrow$ & Slope error $\downarrow$ & Violation rate $\downarrow$ \\", r"\midrule"]
    for k in (2, 3, 4):
        for name, row in extra[str(k)]["weak_physics"].items():
            lines.append(f'{name} & {k} & {row["point"]["mae_logN"]:.3f} & {row["grid_mae_logN"]:.3f} & {row["slope_error"]:.3f} & {row["monotonic_violation"]:.3f} \\\\')
        if k != 4: lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_weak_physics.tex").write_text("\n".join(lines))


def plot_source_aligned():
    ours = json.loads((ROOT / "final_protocol/source_3step_calibrated/metrics.json").read_text())["summary"]
    base = json.loads((ROOT / "final_protocol/deterministic/source_baselines.json").read_text())["known_points"]
    direct = json.loads((ROOT / "final_protocol/deterministic/direct_unet.json").read_text())["known_points"]
    ks = [2, 3, 4]
    methods = ["PA-CBB", "Ridge", "MLP", "Direct U-Net"]
    colors = [BLUE, PURPLE, ROSE, GREY]

    def row(method, k):
        if method == "PA-CBB":
            item = ours[str(k)]
            return item["point"], {"grid_mae_logN": item["grid_mae_logN"], "slope_error": item["slope_error"]}
        if method == "Direct U-Net":
            item = direct[str(k)]
        else:
            item = base[str(k)][method]
        return item["point"], item["curve"]

    fig, axes = plt.subplots(1, 4, figsize=(11.7, 2.9)); x = np.arange(3); width = .18
    specs = [("mae_logN", "Hidden-point MAE", "point", False),
             ("grid_mae_logN", "Full-grid MAE", "curve", False),
             ("slope_error", "Slope error", "curve", False),
             ("factor_2_accuracy", "Factor-of-two accuracy", "point", True)]
    for panel, (key, title, location, _) in enumerate(specs):
        ax = axes[panel]
        for index, method in enumerate(methods):
            values = []
            for k in ks:
                point, curve = row(method, k); values.append((point if location == "point" else curve)[key])
            ax.bar(x + (index - 1.5) * width, values, width, color=colors[index], label=method)
        ax.set_title(title, fontweight="bold"); ax.set_xticks(x, ks); ax.set_xlabel("Observed anchors"); ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Error (log cycles)")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, .94))
    fig.suptitle("AM2022 source holdout under the unified three-step protocol", fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, .86)); save(fig, "source_aligned_comparison")


def lodo_results():
    root = ROOT / "final_protocol/lodo"; rows = []
    for target, label in (("am2022_curves", "AM2022"), ("cma2022_curves", "CMA2022"), ("weld2025_disjoint_curves", "Weld2025")):
        model_path = root / target / "eval/pacbb.json"; base_path = root / target / "eval/nonphysics_baselines.json"
        if not model_path.exists() or not base_path.exists(): return []
        model = json.loads(model_path.read_text()); base = json.loads(base_path.read_text())["external"][target]["known_points"]
        for k in (2, 3, 4):
            best_name, best = min(((name, value["point"]["mae_logN"]) for name, value in base[str(k)].items()), key=lambda pair: pair[1])
            rows.append({"database": label, "known": k, "pacbb": model["known_points"][str(k)]["mae_logN"], "baseline": best, "baseline_name": best_name})
    return rows


def plot_lodo():
    rows = lodo_results()
    if not rows: return False
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=False)
    for ax, database in zip(axes, ("AM2022", "CMA2022", "Weld2025")):
        subset = [row for row in rows if row["database"] == database]; x = np.arange(3); width = .34
        ax.bar(x-width/2, [r["pacbb"] for r in subset], width, color=BLUE, label="PA-CBB")
        ax.bar(x+width/2, [r["baseline"] for r in subset], width, color=ROSE, label="Best non-physics")
        ax.set_title(f"Held out: {database}", fontweight="bold"); ax.set_xticks(x, ["2", "3", "4"]); ax.set_xlabel("Observed anchors"); ax.set_ylabel("Hidden-point MAE"); ax.grid(axis="y", alpha=.22)
    axes[0].legend(frameon=False, fontsize=8); fig.suptitle("True leave-one-database-out evaluation", fontweight="bold", y=1.02); fig.tight_layout(); save(fig, "lodo_comparison")
    lines = [r"\begin{table*}[t]", r"\caption{True leave-one-database-out evaluation. The target database is absent from training and validation. The strongest baseline is selected among ridge, $k$NN, random forest, ExtraTrees, linear, polynomial and PCHIP; the high-cost MLP is retained in the source and pooled studies but not retrained in each LODO fold.}", r"\label{tab:lodo}", r"\centering\scriptsize", r"\begin{tabular}{lccccc}", r"\toprule", r"Held-out database & $k$ & PA-CBB MAE & Best baseline MAE & Baseline & Relative change (\%) \\", r"\midrule"]
    for row in rows:
        change = 100 * (row["pacbb"] - row["baseline"]) / row["baseline"]
        lines.append(f'{row["database"]} & {row["known"]} & {row["pacbb"]:.3f} & {row["baseline"]:.3f} & {row["baseline_name"]} & {change:+.1f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_lodo.tex").write_text("\n".join(lines)); return True


def pooled_results():
    baseline_path = ROOT / "final_protocol/pooled_3step/nonphysics_baselines.json"
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text())["external"]
    rows = []
    for slug, label in (("am2022", "AM2022"), ("cma2022", "CMA2022"),
                        ("hea2022", "HEA2022"), ("nims_derived", "NIMS-derived"),
                        ("weld2025", "Weld2025")):
        model = json.loads((ROOT / f"final_protocol/pooled_3step/{slug}.json").read_text())
        for k in (2, 3, 4):
            candidates = baseline[label]["known_points"][str(k)]
            best_name, best = min(((name, row["point"]["mae_logN"]) for name, row in candidates.items()), key=lambda pair: pair[1])
            rows.append({"database": label, "known": k, "pacbb": model["known_points"][str(k)]["mae_logN"],
                         "baseline": best, "baseline_name": best_name})
    return rows


def plot_pooled():
    rows = pooled_results()
    if not rows:
        return False
    fig, axes = plt.subplots(1, 5, figsize=(12.0, 2.9), sharey=False)
    for ax, database in zip(axes, ("AM2022", "CMA2022", "HEA2022", "NIMS-derived", "Weld2025")):
        subset = [row for row in rows if row["database"] == database]; x = np.arange(3); width = .34
        ax.bar(x-width/2, [r["pacbb"] for r in subset], width, color=BLUE, label="PA-CBB")
        ax.bar(x+width/2, [r["baseline"] for r in subset], width, color=ROSE, label="Best non-physics")
        ax.set_title(database, fontweight="bold"); ax.set_xticks(x, ["2", "3", "4"]); ax.set_xlabel("Anchors")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Hidden-point MAE (log cycles)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, .94))
    fig.suptitle("Aligned pooled evaluation: three steps, 32 samples and identical masks", fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, .89)); save(fig, "pooled_3step_comparison")
    lines = [r"\begin{table*}[t]", r"\caption{Corrected pooled holdout under the unified three-step protocol. The strongest non-physics baseline is selected separately within each database--sparsity task. HEA and NIMS contain only two test curves each and are audit cases, not stable population estimates.}", r"\label{tab:pooled_final}", r"\centering\scriptsize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule", r"Database & $k$ & PA-CBB MAE & Best baseline MAE & Baseline & Relative change (\%) \\", r"\midrule"]
    for row in rows:
        change = 100 * (row["pacbb"] - row["baseline"]) / row["baseline"]
        lines.append(f'{row["database"]} & {row["known"]} & {row["pacbb"]:.3f} & {row["baseline"]:.3f} & {row["baseline_name"]} & {change:+.1f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_pooled_final.tex").write_text("\n".join(lines)); return True


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    OUT.mkdir(parents=True, exist_ok=True); plot_physics_probability(); write_tables(); plot_source_aligned(); plot_pooled(); plot_lodo()
