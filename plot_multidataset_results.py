#!/usr/bin/env python3
"""Create publication figures and tables for the five-database experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "multidataset_results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DATA = ROOT / "external_data"

BLUE = "#3B6C9E"
LIGHT_BLUE = "#8FB3D9"
ROSE = "#CC7A7A"
LIGHT_ROSE = "#E7B7B5"
PURPLE = "#7C6FA8"
GOLD = "#D6A646"
GREY = "#72777F"
PALETTE = [BLUE, LIGHT_BLUE, PURPLE, ROSE, GOLD, "#5B8E7D", "#A97C50", "#9A9EAB", "#C8A2C8"]
METHODS = ["PA-CBB", "kNN", "RandomForest", "ExtraTrees", "MLP", "Linear", "Polynomial", "PCHIP"]
DOMAINS = ["AM2022", "CMA2022", "HEA2022", "Weld2025"]
DOMAIN_FILES = {
    "AM2022": DATA / "pooled_test_domains/pooled_test_am2022_curves.json",
    "CMA2022": DATA / "pooled_test_domains/pooled_test_cma2022_curves.json",
    "HEA2022": DATA / "pooled_test_domains/pooled_test_hea2022_curves.json",
    "Weld2025": DATA / "pooled_test_domains/pooled_test_weld2025_disjoint_curves.json",
}
METRIC_FILES = {d: RESULTS / "pooled_test_model/metrics" / f"{d.lower()}.json" for d in DOMAINS}
PREDICTION_NAMES = {d: d.lower() for d in DOMAINS}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel(ax, label: str) -> None:
    ax.text(-0.10, 1.00, f"({label})", transform=ax.transAxes, ha="right", va="bottom",
            weight="bold", fontsize=9.5, zorder=20)


def save_all(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_results():
    baseline = json.loads((RESULTS / "pooled_test_baselines.json").read_text())
    full = {domain: json.loads(path.read_text()) for domain, path in METRIC_FILES.items()}
    audit = json.loads((DATA / "dataset_audit5.json").read_text())
    zero_shot = {}
    for path in (RESULTS / "am_source_model/metrics").glob("*.json"):
        record = json.loads(path.read_text())
        zero_shot[record["dataset"]] = record
    return baseline, full, audit, zero_shot


def metric_matrix(baseline, full, metric="mae_logN"):
    matrix = np.zeros((len(METHODS), len(DOMAINS), 3))
    for di, domain in enumerate(DOMAINS):
        for ki, known in enumerate((2, 3, 4)):
            matrix[0, di, ki] = full[domain]["known_points"][str(known)][metric]
            for mi, method in enumerate(METHODS[1:], 1):
                matrix[mi, di, ki] = baseline["external"][domain]["known_points"][str(known)][method]["point"][metric]
    return matrix


def figure_baseline_bars(baseline, full) -> None:
    mae = metric_matrix(baseline, full)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5), constrained_layout=True)
    y = np.arange(len(METHODS))
    offsets = [-0.24, 0, 0.24]
    colors = [LIGHT_BLUE, PURPLE, LIGHT_ROSE]
    for di, (ax, domain) in enumerate(zip(axes.flat, DOMAINS)):
        for ki, known in enumerate((2, 3, 4)):
            bars = ax.barh(y + offsets[ki], mae[:, di, ki], height=0.21, color=colors[ki],
                           edgecolor="white", linewidth=0.35, label=f"$k={known}$")
            bars[0].set_color(BLUE)
        ax.invert_yaxis()
        ax.set_yticks(y, METHODS)
        ax.set_xlabel("Hidden-point MAE ($\log_{10}N$; lower is better)")
        ax.set_title(f"{domain} held-out curves")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        best_baseline = mae[1:, di].min(axis=0)
        advantage = 100 * (best_baseline - mae[0, di]) / best_baseline
        ax.text(0.985, 0.975, "PA-CBB gain ($k=2/3/4$): " + ", ".join(f"{v:.1f}%" for v in advantage),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=BLUE, weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5})
        panel(ax, chr(ord("a") + di))
    handles = [Patch(facecolor=color, edgecolor="none", label=f"$k={known}$")
               for color, known in zip(colors, (2, 3, 4))]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Five-database pooled training: complete method versus non-physics baselines",
                 fontsize=12, weight="bold", y=1.02)
    save_all(fig, "multidataset_baseline_bars")


def eligible_points(curve):
    return [(float(p[0]), float(p[1])) for p in curve["points"] if not p[2] and p[0] > 0 and p[1] > 0]


def figure_data_landscape(audit) -> None:
    fig = plt.figure(figsize=(12.2, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])
    rng = np.random.default_rng(20260928)
    for i, domain in enumerate(DOMAINS):
        ax = fig.add_subplot(gs[0, i])
        curves = json.loads(DOMAIN_FILES[domain].read_text())["curves"]
        pts = np.asarray([point for curve in curves for point in eligible_points(curve)], float)
        if len(pts) > 4500:
            pts = pts[rng.choice(len(pts), 4500, replace=False)]
        ax.scatter(np.log10(pts[:, 0]), np.log10(pts[:, 1]), s=7, alpha=0.22,
                   color=PALETTE[i], edgecolors="none", rasterized=True)
        ax.set_title(f"{domain} ($n_c={len(curves)}$)")
        ax.set_xlabel("$\log_{10}$ stress (MPa)")
        if i == 0:
            ax.set_ylabel("$\log_{10}$ cycles to failure")
        ax.grid(color="#E3E3E3", linewidth=0.45)
        panel(ax, chr(ord("a") + i))

    ax_pie = fig.add_subplot(gs[1, 0:2])
    audit_names = ["am2022_curves", "cma2022_curves", "hea2022_curves", "weld2025_disjoint_curves", "nims_derived_curves"]
    labels = ["AM2022", "CMA2022", "HEA2022", "Weld2025", "NIMS-derived"]
    sizes = [audit["datasets"][name]["eligible_curves"] for name in audit_names]
    explode = [0, 0.03, 0.06, 0, 0.12]
    wedges, _ = ax_pie.pie(sizes, startangle=120, colors=PALETTE[:5], explode=explode,
                           wedgeprops={"linewidth": 0.7, "edgecolor": "white"})
    legend_labels = [f"{label}: {size:,} curves" for label, size in zip(labels, sizes)]
    ax_pie.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=False)
    ax_pie.set_title("Eligible-curve composition after harmonization")
    panel(ax_pie, "e")

    ax_box = fig.add_subplot(gs[1, 2:])
    distributions = []
    for name in audit_names:
        source = {
            "am2022_curves": ROOT / "am2022_curves.json",
            "cma2022_curves": DATA / "cma2022_curves.json",
            "hea2022_curves": DATA / "hea2022_curves.json",
            "weld2025_disjoint_curves": DATA / "weld2025_disjoint_curves.json",
            "nims_derived_curves": DATA / "nims_derived_curves.json",
        }[name]
        curves = json.loads(source.read_text())["curves"]
        counts = [len(eligible_points(curve)) for curve in curves if len(eligible_points(curve)) >= 6]
        distributions.append(counts)
    bp = ax_box.boxplot(distributions, tick_labels=labels, patch_artist=True, showfliers=False,
                        medianprops={"color": "#333333", "linewidth": 1.2})
    for patch, color in zip(bp["boxes"], PALETTE[:5]):
        patch.set_facecolor(color); patch.set_alpha(0.72)
    ax_box.set_yscale("log")
    ax_box.set_ylabel("Failure points per curve (log scale)")
    ax_box.set_title("Curve-density distribution")
    ax_box.tick_params(axis="x", rotation=18)
    ax_box.grid(axis="y", color="#E3E3E3", linewidth=0.5)
    panel(ax_box, "f")
    fig.suptitle("Public fatigue data landscape used in the five-database study", fontsize=12, weight="bold")
    save_all(fig, "multidataset_data_landscape")


def load_predictions(domain):
    arrays = []
    for path in sorted((RESULTS / "pooled_test_model/predictions").glob(f"{PREDICTION_NAMES[domain]}_rank*.npz")):
        arrays.append(np.load(path))
    return {key: np.concatenate([arr[key] for arr in arrays]) for key in ("truth", "prediction", "known", "interval_width")}


def figure_predictions_and_boxes(baseline, full) -> None:
    fig = plt.figure(figsize=(12.2, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    rng = np.random.default_rng(11)
    abs_errors = {}
    marker_handles = []
    for i, domain in enumerate(DOMAINS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        pred = load_predictions(domain)
        abs_errors[domain] = {k: np.abs(pred["prediction"][pred["known"] == k] - pred["truth"][pred["known"] == k]) for k in (2, 3, 4)}
        idx = np.arange(len(pred["truth"]))
        if len(idx) > 3500:
            idx = rng.choice(idx, 3500, replace=False)
        for known, color in zip((2, 3, 4), (LIGHT_BLUE, PURPLE, ROSE)):
            keep = idx[pred["known"][idx] == known]
            artist = ax.scatter(pred["truth"][keep], pred["prediction"][keep], color=color,
                                s=8, alpha=0.28, edgecolors="none", rasterized=True,
                                label=f"$k={known}$")
            if i == 0:
                marker_handles.append(artist)
        limits = [min(pred["truth"].min(), pred["prediction"].min()), max(pred["truth"].max(), pred["prediction"].max())]
        ax.plot(limits, limits, linestyle="--", color="#333333", linewidth=0.9)
        ax.set(xlim=limits, ylim=limits, xlabel="Measured $\log_{10}N$", ylabel="Predicted $\log_{10}N$",
               title=f"{domain}: held-out points")
        mean_mae = np.mean([full[domain]["known_points"][str(k)]["mae_logN"] for k in (2, 3, 4)])
        ax.text(0.04, 0.93, f"mean MAE = {mean_mae:.3f}", transform=ax.transAxes, color=BLUE, weight="bold")
        ax.grid(color="#E3E3E3", linewidth=0.45)
        panel(ax, chr(ord("a") + i))
    # Color order is stated in the caption; omitting an in-panel legend keeps
    # annotations and data unobstructed at two-column print size.

    ax_err = fig.add_subplot(gs[0, 2])
    positions, values, colors, tick_pos, tick_labels = [], [], [], [], []
    pos = 1
    for di, domain in enumerate(DOMAINS):
        start = pos
        for known in (2, 3, 4):
            sample = abs_errors[domain][known]
            if len(sample) > 2500:
                sample = sample[rng.choice(len(sample), 2500, replace=False)]
            positions.append(pos); values.append(sample); colors.append(PALETTE[di]); pos += 1
        tick_pos.append(start + 1); tick_labels.append(domain.replace("2022", "").replace("2025", "")); pos += 0.8
    bp = ax_err.boxplot(values, positions=positions, widths=0.72, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 1.1})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.82)
    ax_err.set_xticks(tick_pos, tick_labels)
    ax_err.set_ylabel("Absolute hidden-point error ($\log_{10}N$)")
    ax_err.set_title("PA-CBB point-error distributions\n(three boxes per domain: $k=2,3,4$)")
    ax_err.grid(axis="y", color="#E3E3E3", linewidth=0.5)
    panel(ax_err, "e")

    ax_task = fig.add_subplot(gs[1, 2])
    mae = metric_matrix(baseline, full)
    task_values = [mae[mi].reshape(-1) for mi in range(len(METHODS))]
    bp = ax_task.boxplot(task_values, vert=False, tick_labels=METHODS, patch_artist=True, showfliers=False,
                         medianprops={"color": "white", "linewidth": 1.1})
    for mi, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(BLUE if mi == 0 else GREY); patch.set_alpha(0.9 if mi == 0 else 0.55)
    ax_task.set_xlabel("MAE across 12 domain--sparsity tasks")
    ax_task.set_title("Cross-task robustness of all methods")
    ax_task.invert_yaxis()
    ax_task.grid(axis="x", color="#E3E3E3", linewidth=0.5)
    panel(ax_task, "f")
    fig.suptitle("Held-out prediction fidelity and error distributions", fontsize=12, weight="bold", y=1.02)
    save_all(fig, "multidataset_predictions_boxes")


def figure_heatmaps(baseline, full, zero_shot) -> None:
    mae = metric_matrix(baseline, full)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), constrained_layout=True)
    tasks = [f"{d.replace('2022','').replace('2025','')}\n$k={k}$" for d in DOMAINS for k in (2, 3, 4)]
    image = axes[0, 0].imshow(mae.reshape(len(METHODS), -1), aspect="auto", cmap="RdPu", vmin=0.15, vmax=np.quantile(mae, 0.92))
    axes[0, 0].set_yticks(range(len(METHODS)), METHODS)
    axes[0, 0].set_xticks(range(len(tasks)), tasks, rotation=45, ha="right")
    axes[0, 0].set_title("Hidden-point MAE across methods and tasks")
    for j in range(len(tasks)):
        axes[0, 0].text(j, 0, f"{mae[0].reshape(-1)[j]:.2f}", ha="center", va="center", color="#233142", weight="bold", fontsize=6.3)
    fig.colorbar(image, ax=axes[0, 0], label="MAE ($\log_{10}N$)", shrink=0.82)
    panel(axes[0, 0], "a")

    best = mae[1:].min(axis=0)
    advantage = 100 * (best - mae[0]) / best
    im = axes[0, 1].imshow(advantage, cmap=LinearSegmentedColormap.from_list("gain", ["#F4E7E7", LIGHT_BLUE, BLUE]), vmin=0, vmax=55, aspect="auto")
    axes[0, 1].set_yticks(range(len(DOMAINS)), DOMAINS)
    axes[0, 1].set_xticks(range(3), ["$k=2$", "$k=3$", "$k=4$"])
    axes[0, 1].set_title("MAE reduction versus strongest baseline")
    for i in range(len(DOMAINS)):
        for j in range(3):
            axes[0, 1].text(j, i, f"{advantage[i, j]:.1f}%", ha="center", va="center", weight="bold",
                            color="white" if advantage[i, j] > 34 else "#30343B")
    fig.colorbar(im, ax=axes[0, 1], label="Relative reduction (%)", shrink=0.82)
    panel(axes[0, 1], "b")

    metrics = ["r2", "factor_2_accuracy", "coverage90", "monotonic_violation"]
    labels = ["$R^2$", "Factor-2", "Raw cov.90", "Monotonic viol."]
    summary = np.zeros((len(DOMAINS), len(metrics)))
    for di, domain in enumerate(DOMAINS):
        for mi, metric in enumerate(metrics):
            summary[di, mi] = np.mean([full[domain]["known_points"][str(k)][metric] for k in (2, 3, 4)])
    im2 = axes[1, 0].imshow(summary, cmap="PuBu", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set_yticks(range(len(DOMAINS)), DOMAINS)
    axes[1, 0].set_xticks(range(len(labels)), labels)
    axes[1, 0].set_title("Complete-model diagnostics (averaged over $k$)")
    for i in range(len(DOMAINS)):
        for j in range(len(metrics)):
            axes[1, 0].text(j, i, f"{summary[i,j]:.2f}", ha="center", va="center",
                            color="white" if summary[i, j] > 0.55 else "#30343B", fontsize=7.5)
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.82)
    panel(axes[1, 0], "c")

    common = ["CMA2022", "HEA2022", "Weld2025"]
    transfer = np.zeros((len(common), 3))
    for di, domain in enumerate(common):
        for ki, known in enumerate((2, 3, 4)):
            z = zero_shot[domain]["known_points"][str(known)]["mae_logN"]
            p = full[domain]["known_points"][str(known)]["mae_logN"]
            transfer[di, ki] = 100 * (z - p) / z
    im3 = axes[1, 1].imshow(transfer, cmap=LinearSegmentedColormap.from_list("transfer", [LIGHT_ROSE, "#F7F7F7", BLUE]), vmin=-10, vmax=80, aspect="auto")
    axes[1, 1].set_yticks(range(len(common)), common)
    axes[1, 1].set_xticks(range(3), ["$k=2$", "$k=3$", "$k=4$"])
    axes[1, 1].set_title("Benefit of pooled training over AM-only zero-shot")
    for i in range(len(common)):
        for j in range(3):
            axes[1, 1].text(j, i, f"{transfer[i,j]:.1f}%", ha="center", va="center", weight="bold",
                            color="white" if transfer[i, j] > 45 else "#30343B")
    fig.colorbar(im3, ax=axes[1, 1], label="MAE reduction (%)", shrink=0.82)
    panel(axes[1, 1], "d")
    fig.suptitle("Cross-database performance, advantage and transfer diagnostics", fontsize=12, weight="bold")
    save_all(fig, "multidataset_heatmaps")


def write_tables(baseline, full) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in DOMAINS:
        for known in (2, 3, 4):
            row = {"Dataset": domain, "k": known}
            row["PA-CBB"] = full[domain]["known_points"][str(known)]["mae_logN"]
            for method in METHODS[1:]:
                row[method] = baseline["external"][domain]["known_points"][str(known)][method]["point"]["mae_logN"]
            candidates = {method: row[method] for method in METHODS[1:]}
            best_name, best_value = min(candidates.items(), key=lambda item: item[1])
            row["Best baseline"] = best_name
            row["Relative MAE reduction (%)"] = 100 * (best_value - row["PA-CBB"]) / best_value
            row["PA-CBB R2"] = full[domain]["known_points"][str(known)]["r2"]
            row["PA-CBB Factor-2"] = full[domain]["known_points"][str(known)]["factor_2_accuracy"]
            rows.append(row)
    fields = ["Dataset", "k"] + METHODS + ["Best baseline", "Relative MAE reduction (%)", "PA-CBB R2", "PA-CBB Factor-2"]
    with (TABLES / "main_five_dataset_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Main held-out comparison after pooled multi-database training. Values are hidden-point MAE in $\log_{10}N$; lower is better. Bold denotes the best result in each row. Basquin-only and bridge-only variants are reserved for ablation.}",
        r"\label{tab:five_dataset_main}",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"Dataset & $k$ & PA-CBB & Ridge & kNN & RF & ET & MLP & Linear & Poly. & PCHIP & Gain (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        numeric = {method: row[method] for method in METHODS}
        winner = min(numeric, key=numeric.get)
        values = []
        for method in METHODS:
            value = f"{numeric[method]:.3f}"
            values.append(r"\textbf{" + value + "}" if method == winner else value)
        lines.append(f"{row['Dataset']} & {row['k']} & " + " & ".join(values)
                     + f" & {row['Relative MAE reduction (%)']:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "main_five_dataset_results.tex").write_text("\n".join(lines))

    summary = {
        "tasks": len(rows),
        "pa_cbb_wins": sum(row["PA-CBB"] == min(row[method] for method in METHODS) for row in rows),
        "mean_pa_cbb_mae": float(np.mean([row["PA-CBB"] for row in rows])),
        "mean_best_baseline_mae": float(np.mean([min(row[m] for m in METHODS[1:]) for row in rows])),
        "mean_relative_reduction_percent": float(np.mean([row["Relative MAE reduction (%)"] for row in rows])),
        "min_relative_reduction_percent": float(np.min([row["Relative MAE reduction (%)"] for row in rows])),
        "max_relative_reduction_percent": float(np.max([row["Relative MAE reduction (%)"] for row in rows])),
    }
    (TABLES / "main_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    style()
    baseline, full, audit, zero_shot = load_results()
    figure_baseline_bars(baseline, full)
    figure_data_landscape(audit)
    figure_predictions_and_boxes(baseline, full)
    figure_heatmaps(baseline, full, zero_shot)
    write_tables(baseline, full)


if __name__ == "__main__":
    main()
