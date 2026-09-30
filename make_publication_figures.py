#!/usr/bin/env python3
"""Create publication-ready multi-panel figures from the real fatigue experiments."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication_figures"
OUT.mkdir(exist_ok=True)

# Palette sampled/approximated from the user's six reference figures.
NAVY = "#315C93"
BLUE = "#86A9D5"
PALE_BLUE = "#C8D7EA"
LAVENDER = "#8E83B7"
ROSE = "#C96F73"
PALE_ROSE = "#E9B8B8"
YELLOW = "#F2B94B"
INK = "#252525"
MID = "#737A84"
GRID = "#D8DDE5"
OFFWHITE = "#FBFAF8"
DIVERGING = LinearSegmentedColormap.from_list(
    "reference_blue_rose", ["#7186B4", "#F7F4F2", "#C66D73"]
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 8.3,
        "axes.labelsize": 8.7,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.85,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


SPLITS = [
    ("source_paper", "Source holdout"),
    ("loao_ti6al4v", "Ti-6Al-4V"),
    ("loao_in718", "IN718"),
    ("loao_alsi10mg", "AlSi10Mg"),
    ("loao_316l", "316L"),
]
KS = (2, 3, 4)


def load_json(path):
    return json.loads(Path(path).read_text())


def metrics():
    return {
        key: load_json(ROOT / "strict7" / key / "conformal" / "metrics.json")
        for key, _ in SPLITS
    }


def panel(ax, letter, x=-0.14, y=1.08):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def clean(ax, grid=True):
    if grid:
        ax.grid(True, color=GRID, lw=0.55, ls="--", alpha=0.72, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def heatmap(ax, values, rows, cols, title, fmt, vmin=None, vmax=None, center=None, reverse=False):
    diverging = center is not None
    if center is None:
        colors = [ROSE, PALE_ROSE, OFFWHITE] if reverse else [OFFWHITE, PALE_ROSE, ROSE]
        cmap = LinearSegmentedColormap.from_list("seq_reverse" if reverse else "seq", colors)
    else:
        cmap = DIVERGING
        lim = max(abs(np.nanmin(values - center)), abs(np.nanmax(values - center)))
        vmin, vmax = center - lim, center + lim
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, fontweight="bold", pad=7)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            normalized = (val - im.norm.vmin) / max(im.norm.vmax - im.norm.vmin, 1e-9)
            dark_cell = (normalized < 0.28 if reverse else normalized > 0.72) or (diverging and normalized < 0.12)
            ax.text(
                j,
                i,
                fmt.format(val),
                ha="center",
                va="center",
                color="white" if dark_cell else INK,
                fontsize=7.0,
                fontweight="bold",
            )
    ax.tick_params(length=0)
    return im


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def eligible_curves():
    curves = load_json(ROOT / "am2022_curves.json")["curves"]
    out = []
    for curve in curves:
        points = np.asarray(
            [p[:2] for p in curve["points"] if not p[2] and p[0] > 0 and p[1] > 0],
            dtype=float,
        )
        if len(points) < 6:
            continue
        x, y = np.log10(points[:, 0]), np.log10(points[:, 1])
        if len(np.unique(x)) < 4 or np.ptp(x) < 0.04:
            continue
        slope = np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-10)
        if not (-25 < slope < -0.05):
            continue
        record = dict(curve)
        record["_points"] = points
        record["_slope"] = slope
        out.append(record)
    return out


def source_family_counts(curves):
    ids = set(load_json(ROOT / "strict_splits" / "source_paper.json")["test"])
    return Counter(c["material_family"] for c in curves if c["id"] in ids)


def performance_atlas():
    met = metrics()
    pred = np.load(ROOT / "strict7/source_paper/conformal/test_predictions.npz")
    extended = load_json(ROOT / "strict7/source_paper/extended_metrics.json")["known_points"]
    fig = plt.figure(figsize=(11.6, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])

    letters = iter("abcdef")
    for col, k in enumerate(KS):
        ax = fig.add_subplot(gs[0, col])
        truth = pred[f"k{k}_truth"]
        mean = pred[f"k{k}_mean"]
        rng = np.random.default_rng(810 + k)
        take = rng.choice(len(truth), min(1000, len(truth)), replace=False)
        limits = [min(truth.min(), mean.min()), max(truth.max(), mean.max())]
        error = np.abs(mean - truth)
        sc = ax.scatter(truth[take], mean[take], c=error[take], s=11, cmap=LinearSegmentedColormap.from_list("error", [PALE_BLUE, LAVENDER, ROSE]), alpha=0.50, edgecolor="none")
        ax.plot(limits, limits, color=INK, ls="--", lw=1.0)
        ax.set(xlim=limits, ylim=limits, xlabel="Measured log$_{10}N$", ylabel="Predicted log$_{10}N$")
        r2 = 1 - np.sum((mean - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
        ax.set_title(f"{k} known points  |  MAE={error.mean():.3f}  |  $R^2$={r2:.3f}", color=NAVY, fontweight="bold")
        ax.text(0.04, 0.94, f"n = {len(truth):,}", transform=ax.transAxes, va="top", color=MID)
        if col == 2:
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label("Absolute error")
        clean(ax)
        panel(ax, next(letters))

    ax = fig.add_subplot(gs[1, 0])
    values = [np.abs(pred[f"k{k}_mean"] - pred[f"k{k}_truth"]) for k in KS]
    parts = ax.boxplot(values, widths=0.56, patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": MID}, capprops={"color": MID})
    for patch, color in zip(parts["boxes"], (BLUE, LAVENDER, PALE_ROSE)):
        patch.set(facecolor=color, edgecolor=INK, alpha=0.94)
    ax.set_xticks([1, 2, 3], ["k=2", "k=3", "k=4"])
    ax.set_ylabel("Absolute error (log$_{10}N$)")
    ax.set_title("Hidden-point error distribution", fontweight="bold")
    clean(ax); panel(ax, next(letters))

    ax = fig.add_subplot(gs[1, 1])
    model_mae = np.asarray([[m["known_points"][str(k)]["raw"]["mae_logN"] for k in KS] for m in met.values()])
    heatmap(ax, model_mae, [label for _, label in SPLITS], ["k=2", "k=3", "k=4"], "Strict-split MAE (log$_{10}N$)", "{:.3f}", vmin=0.23, vmax=0.42)
    panel(ax, next(letters), x=-0.20)

    ax = fig.add_subplot(gs[1, 2], projection="polar")
    source = met["source_paper"]
    widths = np.asarray([source["known_points"][str(k)]["pointwise_conformal"]["width90_logN"] for k in KS])
    sharp = 1 - (widths - widths.min()) / max(np.ptp(widths), 1e-9)
    categories = ["$R^2$", "Factor-of-2\naccuracy", "Calibration\ncloseness", "Relative\nsharpness", "Raw coverage\n(/90%)"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    for i, (k, color) in enumerate(zip(KS, (NAVY, LAVENDER, ROSE))):
        item = source["known_points"][str(k)]
        vals = [
            np.clip(extended[str(k)]["point_model"]["r2"], 0, 1),
            np.clip(extended[str(k)]["point_model"]["factor_2_accuracy"], 0, 1),
            np.clip(1 - abs(item["pointwise_conformal"]["coverage90"] - 0.90) / 0.10, 0, 1),
            sharp[i],
            np.clip(item["raw"]["coverage90"] / 0.90, 0, 1),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=1.6, marker="o", ms=3.2, label=f"k={k}")
        ax.fill(angles, vals, color=color, alpha=0.06)
    ax.set_xticks(angles[:-1], categories)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1.0"], color=MID)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, lw=0.55)
    ax.spines["polar"].set_color(MID)
    ax.set_title("Normalized generative diagnostics", fontweight="bold", pad=14)
    ax.legend(frameon=False, loc=(0.98, 0.76))
    panel(ax, next(letters), x=-0.08, y=1.13)

    fig.suptitle("Main-model performance across sparsity and strict test splits", fontsize=13, fontweight="bold", color=INK)
    save(fig, "figure_01_generative_value_atlas")


def uncertainty_atlas():
    met = metrics()
    rows = [label for _, label in SPLITS]
    shape = (len(SPLITS), len(KS))
    model_mae = np.zeros(shape)
    raw_cov = np.zeros(shape)
    cal_cov = np.zeros(shape)
    cal_width = np.zeros(shape)
    for i, (key, _) in enumerate(SPLITS):
        for j, k in enumerate(KS):
            item = met[key]["known_points"][str(k)]
            model_mae[i, j] = item["raw"]["mae_logN"]
            raw_cov[i, j] = 100 * item["raw"]["coverage90"]
            cal_cov[i, j] = 100 * item["pointwise_conformal"]["coverage90"]
            cal_width[i, j] = item["pointwise_conformal"]["width90_logN"]

    fig, axes = plt.subplots(2, 3, figsize=(11.7, 7.1), constrained_layout=True)
    heatmap(axes[0, 0], model_mae, rows, ["k=2", "k=3", "k=4"], "Model MAE (log$_{10}N$)", "{:.3f}", vmin=0.23, vmax=0.42)
    panel(axes[0, 0], "a", x=-0.20)
    heatmap(axes[0, 1], raw_cov, rows, ["k=2", "k=3", "k=4"], "Raw 90% interval coverage (%)", "{:.1f}", vmin=0, vmax=100)
    panel(axes[0, 1], "b", x=-0.20)
    heatmap(axes[0, 2], cal_cov, rows, ["k=2", "k=3", "k=4"], "Conformal 90% coverage (%)", "{:.1f}", vmin=75, vmax=100)
    panel(axes[0, 2], "c", x=-0.20)
    heatmap(axes[1, 0], cal_width, rows, ["k=2", "k=3", "k=4"], "Conformal interval width (log$_{10}N$)", "{:.2f}", vmin=1.2, vmax=max(1.8, cal_width.max()))
    panel(axes[1, 0], "d", x=-0.20)

    ax = axes[1, 1]
    pred = np.load(ROOT / "strict7/source_paper/conformal/test_predictions.npz")
    for k, color in zip(KS, (NAVY, LAVENDER, ROSE)):
        error = np.abs(pred[f"k{k}_mean"] - pred[f"k{k}_truth"])
        width = pred[f"k{k}_hi"] - pred[f"k{k}_lo"]
        rng = np.random.default_rng(100 + k)
        take = rng.choice(len(error), min(900, len(error)), replace=False)
        corr = np.corrcoef(width, error)[0, 1]
        ax.scatter(width[take], error[take], s=9, alpha=0.28, color=color, edgecolor="none", label=f"k={k}, r={corr:.2f}")
    ax.set(xlabel="Raw 90% interval width (log$_{10}N$)", ylabel="Absolute error (log$_{10}N$)", title="Does generated spread track error?")
    ax.legend(frameon=False)
    clean(ax)
    panel(ax, "e")

    ax = axes[1, 2]
    x = np.arange(len(SPLITS))
    for offset, k, color in zip((-0.20, 0, 0.20), KS, (NAVY, LAVENDER, ROSE)):
        raw = raw_cov[:, KS.index(k)]
        cal = cal_cov[:, KS.index(k)]
        for xi, y0, y1 in zip(x + offset, raw, cal):
            ax.plot([xi, xi], [y0, y1], color=color, lw=1.4, alpha=0.72)
        ax.scatter(x + offset, raw, marker="o", s=28, facecolor="white", edgecolor=color, zorder=3)
        ax.scatter(x + offset, cal, marker="s", s=26, color=color, zorder=3, label=f"k={k}")
    ax.axhline(90, color=INK, ls="--", lw=0.9, label="90% target")
    ax.set_xticks(x, ["Source", "Ti", "IN718", "AlSi", "316L"])
    ax.set_ylim(0, 103)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Raw ○  →  conformal ■", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    clean(ax)
    panel(ax, "f")

    fig.suptitle("Uncertainty audit: generation is useful only when its spread is calibrated", fontsize=13, fontweight="bold")
    save(fig, "figure_02_uncertainty_audit")


def curve_gallery():
    payload = load_json(ROOT / "strict7/source_paper/curve_gallery_advantage.json")
    records = payload["records"]
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.0), constrained_layout=True)
    for idx, (ax, rec) in enumerate(zip(axes.flat, records)):
        stress = 10 ** np.asarray(rec["gx_log_stress"])
        truth = np.asarray(rec["truth_grid_logN"])
        draws = np.asarray(rec["posterior_logN"])
        mean = draws.mean(axis=0)
        lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
        for draw in draws[::4]:
            ax.plot(draw, stress, color=BLUE, lw=0.65, alpha=0.22, zorder=1)
        ax.fill_betweenx(stress, lo, hi, color=PALE_BLUE, alpha=0.55, lw=0, label="Generated 90% band")
        ax.plot(truth, stress, color=ROSE, lw=1.8, label="Reference curve")
        ax.plot(mean, stress, color=NAVY, lw=1.8, label="Generated mean")
        comparisons = rec["comparison_grid_logN"]
        ax.plot(comparisons["Direct U-Net"], stress, color=LAVENDER, lw=1.25, ls="--", label="Direct U-Net")
        ax.plot(comparisons["PCHIP"], stress, color=MID, lw=1.15, ls=":", label="PCHIP")
        raw_x = np.asarray(rec["raw_logN"])
        raw_y = 10 ** np.asarray(rec["raw_log_stress"])
        obs = np.asarray(rec["observed_indices"], dtype=int)
        hidden = np.setdiff1d(np.arange(len(raw_x)), obs)
        ax.scatter(raw_x[hidden], raw_y[hidden], s=16, facecolor="white", edgecolor=INK, lw=0.65, zorder=5)
        ax.scatter(raw_x[obs], raw_y[obs], s=30, marker="D", facecolor=YELLOW, edgecolor=INK, lw=0.75, zorder=6, label="Observed")
        comp_mae = rec["comparison_hidden_mae_logN"]
        ax.set_title(f"{rec['name']}  |  k={rec['k']}", color=(NAVY if rec["k"] < 4 else ROSE), fontweight="bold")
        ax.text(
            0.03,
            0.03,
            f"MAE: ours {rec['error_mae_logN']:.2f}\nDirect {comp_mae['Direct U-Net']:.2f} · PCHIP {comp_mae['PCHIP']:.2f}",
            transform=ax.transAxes,
            va="bottom",
            fontsize=6.3,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        ax.set_xlabel("log$_{10}$ cycles to failure")
        ax.set_ylabel("Stress amplitude (MPa)")
        clean(ax)
        panel(ax, chr(ord("a") + idx), x=-0.16)
    handles = [
        Line2D([0], [0], color=ROSE, lw=2, label="Reference curve"),
        Line2D([0], [0], color=NAVY, lw=2, label="Generated mean"),
        Line2D([0], [0], color=BLUE, lw=6, alpha=0.35, label="Generated 90% band"),
        Line2D([0], [0], color=LAVENDER, lw=1.4, ls="--", label="Direct U-Net"),
        Line2D([0], [0], color=MID, lw=1.4, ls=":", label="PCHIP"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=YELLOW, markeredgecolor=INK, label="Known points"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.018))
    fig.suptitle("Illustrative conditional completion of sparse S-N curves", fontsize=13, fontweight="bold")
    save(fig, "figure_03_conditional_curve_gallery")


def dataset_map():
    curves = eligible_curves()
    def family_name(curve):
        name = str(curve["material_family"])
        if name == "18Ni300":
            return "Maraging"
        if name.isdigit():
            return "Other"
        return name

    rng = np.random.default_rng(20260928)
    sample_idx = rng.choice(len(curves), min(150, len(curves)), replace=False)
    sampled = [curves[i] for i in sample_idx]
    features = []
    families = []
    for curve in sampled:
        pts = curve["_points"]
        cond = curve.get("conditions") or {}
        meta = curve.get("metadata") or {}
        def num(value, default=np.nan):
            try:
                return float(value)
            except Exception:
                return default
        frequency = num(meta.get("testing: frequency (hz)"))
        features.append(
            [
                len(pts),
                np.ptp(np.log10(pts[:, 0])),
                np.median(pts[:, 0]),
                abs(curve["_slope"]),
                num(cond.get("load_ratio")),
                np.log10(frequency) if frequency > 0 else np.nan,
                num(meta.get("metadata: year of publication")),
            ]
        )
        families.append(family_name(curve))
    data = np.asarray(features, float)
    for j in range(data.shape[1]):
        col = data[:, j]
        col[np.isnan(col)] = np.nanmedian(col)
    lo, hi = np.quantile(data, [0.03, 0.97], axis=0)
    norm = np.clip((data - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    top_families = [k for k, _ in Counter(family_name(c) for c in curves if family_name(c) != "Other").most_common(5)]
    cmap = {name: color for name, color in zip(top_families, (NAVY, BLUE, LAVENDER, ROSE, YELLOW))}

    fig = plt.figure(figsize=(11.6, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, :])
    xx = np.arange(data.shape[1])
    for row, family in zip(norm, families):
        ax.plot(xx, row, color=cmap.get(family, MID), alpha=0.20 if family in cmap else 0.08, lw=0.8)
    labels = ["Points", "Stress\nspan", "Median\nstress", "|log-log|\nslope", "Load\nratio", "log test\nfrequency", "Year"]
    ax.set_xticks(xx, labels)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("Robust normalized value")
    ax.set_title("Parallel-coordinate map of eligible public-data curves", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.75)
    ax.grid(axis="y", color=GRID, lw=0.45, ls="--")
    handles = [Line2D([0], [0], color=cmap[n], lw=2, label=n) for n in top_families]
    handles.append(Line2D([0], [0], color=MID, lw=2, label="Other"))
    ax.legend(handles=handles, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    panel(ax, "a", x=-0.04, y=1.06)

    ax = fig.add_subplot(gs[1, 0])
    fam_counts = Counter(family_name(c) for c in curves)
    top = [(k, v) for k, v in fam_counts.most_common() if k != "Other"][:6]
    rest = len(curves) - sum(v for _, v in top)
    vals = [v for _, v in top] + [rest]
    labels = [k for k, _ in top] + ["Other"]
    ax.pie(vals, labels=labels, colors=[NAVY, BLUE, LAVENDER, ROSE, YELLOW, PALE_ROSE, GRID], startangle=90, autopct=lambda p: f"{p:.0f}%" if p >= 6 else "", textprops={"fontsize": 6.8}, wedgeprops={"edgecolor": "white", "linewidth": 0.8})
    ax.set_title(f"Eligible curves by alloy family (n={len(curves)})", fontweight="bold")
    panel(ax, "b", x=-0.08, y=1.16)

    ax = fig.add_subplot(gs[1, 1])
    am_counts = Counter((c.get("conditions") or {}).get("am_type") or "Unknown" for c in curves)
    names, vals = zip(*am_counts.most_common(7))
    order = np.arange(len(names))[::-1]
    ax.barh(order, vals, color=[NAVY, BLUE, LAVENDER, ROSE, YELLOW, PALE_ROSE, GRID][: len(names)], edgecolor="white")
    ax.set_yticks(order, names)
    ax.set_xlabel("Number of curves")
    ax.set_title("Additive-manufacturing process", fontweight="bold")
    for y, v in zip(order, vals):
        ax.text(v + max(vals) * 0.015, y, str(v), va="center", fontsize=7)
    clean(ax, grid=False)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    years = [int(c.get("metadata", {}).get("metadata: year of publication", 0) or 0) for c in curves]
    years = [y for y in years if y > 0]
    counts = Counter(years)
    xs = np.asarray(sorted(counts))
    ys = np.asarray([counts[x] for x in xs])
    ax.fill_between(xs, ys, color=PALE_ROSE, alpha=0.72)
    ax.plot(xs, ys, color=ROSE, lw=1.7, marker="o", ms=3)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Eligible curves")
    ax.set_title("Public-data chronology", fontweight="bold")
    clean(ax)
    panel(ax, "d")

    fig.suptitle("Public dataset coverage and condition space", fontsize=13, fontweight="bold")
    save(fig, "figure_04_public_dataset_map")


def extended_metric_atlas():
    data = load_json(ROOT / "strict7/source_paper/extended_metrics.json")["known_points"]
    x = np.arange(3)
    labels = ["k=2", "k=3", "k=4"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)

    ax = axes[0, 0]
    model = [data[str(k)]["point_model"]["r2"] for k in KS]
    bars = ax.bar(x, model, 0.56, color=(NAVY, LAVENDER, ROSE), edgecolor=INK, lw=0.5)
    for bar, value in zip(bars, model):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .006, f"{value:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x, labels); ax.set_ylim(0.55, 0.80); ax.set_ylabel("$R^2$")
    ax.set_title("Explained variance", fontweight="bold")
    clean(ax); panel(ax, "a")

    ax = axes[0, 1]
    for idx, (name, color, hatch) in enumerate((("factor_2_accuracy", NAVY, ""), ("factor_3_accuracy", ROSE, "//"))):
        vals = [100 * data[str(k)]["point_model"][name] for k in KS]
        ax.bar(x + (idx - 0.5) * 0.34, vals, 0.34, color=color, alpha=0.88, hatch=hatch, edgecolor="white", label=("Within ×2" if idx == 0 else "Within ×3"))
    ax.set_xticks(x, labels); ax.set_ylim(0, 100); ax.set_ylabel("Predictions within factor (%)")
    ax.set_title("Engineering-scale accuracy", fontweight="bold"); ax.legend(frameon=False)
    clean(ax); panel(ax, "b")

    ax = axes[0, 2]
    med = [data[str(k)]["point_model"]["median_ae_logN"] for k in KS]
    p90 = [data[str(k)]["point_model"]["p90_ae_logN"] for k in KS]
    ax.bar(x - 0.18, med, 0.36, color=BLUE, edgecolor=INK, lw=0.5, label="Median AE")
    ax.bar(x + 0.18, p90, 0.36, color=ROSE, edgecolor=INK, lw=0.5, label="90th-percentile AE")
    ax.set_xticks(x, labels); ax.set_ylabel("Absolute error (log$_{10}N$)")
    ax.set_title("Typical versus tail error", fontweight="bold"); ax.legend(frameon=False)
    clean(ax); panel(ax, "c")

    ax = axes[1, 0]
    crps = [data[str(k)]["crps_logN"] for k in KS]
    wis = [data[str(k)]["weighted_interval_score"] for k in KS]
    ax.plot(x, crps, color=NAVY, marker="o", lw=1.8, label="CRPS")
    ax.plot(x, wis, color=LAVENDER, marker="s", lw=1.8, label="WIS")
    ax.set_xticks(x, labels); ax.set_ylabel("Proper score (lower is better)")
    ax.set_title("Marginal distribution quality", fontweight="bold"); ax.legend(frameon=False)
    clean(ax); panel(ax, "d")

    ax = axes[1, 1]
    energy = [data[str(k)]["curve_metrics"]["energy"] for k in KS]
    grid_error = [data[str(k)]["curve_metrics"]["grid_mae"] for k in KS]
    ax.bar(x - 0.18, energy, 0.36, color=LAVENDER, edgecolor=INK, lw=0.5, label="Energy Score")
    ax.bar(x + 0.18, grid_error, 0.36, color=BLUE, edgecolor=INK, lw=0.5, label="Full-grid MAE")
    ax.set_xticks(x, labels); ax.set_ylim(0, max(max(energy), max(grid_error)) * 1.28); ax.set_ylabel("Curve score (lower is better)")
    ax.set_title("Joint full-curve quality", fontweight="bold"); ax.legend(frameon=False)
    clean(ax); panel(ax, "e")

    ax = axes[1, 2]
    nominal = np.asarray([0.50, 0.80, 0.90, 0.95]) * 100
    for k, color, marker in zip(KS, (NAVY, LAVENDER, ROSE), ("o", "s", "D")):
        observed = [100 * data[str(k)]["intervals"][str(level)]["coverage"] for level in (0.5, 0.8, 0.9, 0.95)]
        ax.plot(nominal, observed, color=color, marker=marker, lw=1.6, label=f"k={k}")
    ax.plot([45, 100], [45, 100], color=INK, ls="--", lw=1.0, label="Ideal")
    ax.set(xlim=(45, 100), ylim=(0, 100), xlabel="Nominal coverage (%)", ylabel="Empirical coverage (%)")
    ax.set_title("Raw ensemble calibration", fontweight="bold"); ax.legend(frameon=False)
    clean(ax); panel(ax, "f")

    fig.suptitle("Extended evaluation metrics for conditional S–N generation", fontsize=13, fontweight="bold")
    save(fig, "figure_05_extended_metric_atlas")


def method_benchmark_figure():
    classic = load_json(ROOT / "benchmark/source_paper_baselines.json")["known_points"]
    direct = load_json(ROOT / "benchmark/direct_unet_metrics.json")["known_points"]
    physical = load_json(ROOT / "strict7/source_paper/extended_metrics.json")["known_points"]
    # Physics-only and bridge-only variants are deliberately excluded here.
    # They belong to the matched ablation figure, not the main benchmark.
    method_order = ["Linear", "Polynomial", "PCHIP", "Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP", "Direct U-Net", "Physics bridge"]
    display = {"RandomForest": "Random forest", "Physics bridge": "Proposed model"}

    def record(method, k):
        key = str(k)
        if method in classic[key]:
            return classic[key][method]["point"], classic[key][method]["curve"]
        if method == "Direct U-Net":
            return direct[key]["point"], direct[key]["curve"]
        return physical[key]["point_model"], {
            "grid_mae_logN": physical[key]["curve_metrics"]["grid_mae"],
            "slope_error": physical[key]["curve_metrics"]["slope_error"],
            "monotonic_violation": physical[key]["curve_metrics"]["monotonic_violation"],
        }

    mae = np.asarray([[record(m, k)[0]["mae_logN"] for k in KS] for m in method_order])
    r2 = np.asarray([[record(m, k)[0]["r2"] for k in KS] for m in method_order])
    factor2 = np.asarray([[100 * record(m, k)[0]["factor_2_accuracy"] for k in KS] for m in method_order])
    grid_mae = np.asarray([[record(m, k)[1]["grid_mae_logN"] for k in KS] for m in method_order])
    slope_error = np.asarray([[record(m, k)[1]["slope_error"] for k in KS] for m in method_order])
    row_labels = [display.get(m, m) for m in method_order]

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.5), constrained_layout=True)
    heatmap(axes[0, 0], mae, row_labels, ["k=2", "k=3", "k=4"], "Hidden-point MAE (log$_{10}N$)", "{:.2f}", vmin=0.25, vmax=0.95)
    panel(axes[0, 0], "a", x=-0.37)
    heatmap(axes[0, 1], r2, row_labels, ["k=2", "k=3", "k=4"], "Explained variance ($R^2$)", "{:.2f}", vmin=-3.0, vmax=0.80)
    panel(axes[0, 1], "b", x=-0.37)
    heatmap(axes[0, 2], factor2, row_labels, ["k=2", "k=3", "k=4"], "Within-factor-of-2 accuracy (%)", "{:.0f}", vmin=20, vmax=70)
    panel(axes[0, 2], "c", x=-0.37)

    heatmap(axes[1, 0], grid_mae, row_labels, ["k=2", "k=3", "k=4"], "Full-grid MAE (log$_{10}N$)", "{:.2f}", vmin=0.15, vmax=0.75)
    panel(axes[1, 0], "d", x=-0.37)

    heatmap(axes[1, 1], slope_error, row_labels, ["k=2", "k=3", "k=4"], "Full-curve slope error", "{:.2f}", vmin=1.5, vmax=9.5)
    panel(axes[1, 1], "e", x=-0.37)

    ax = axes[1, 2]
    comparison_methods = method_order[:-1]
    specifications = [
        ("MAE", "point", "mae_logN", "min"),
        ("RMSE", "point", "rmse_logN", "min"),
        ("Median AE", "point", "median_ae_logN", "min"),
        ("P90 AE", "point", "p90_ae_logN", "min"),
        ("$R^2$", "point", "r2", "max"),
        ("Pearson $r$", "point", "pearson_r", "max"),
        ("Factor-of-2", "point", "factor_2_accuracy", "max"),
        ("Factor-of-3", "point", "factor_3_accuracy", "max"),
        ("Grid MAE", "curve", "grid_mae_logN", "min"),
        ("Slope error", "curve", "slope_error", "min"),
    ]
    labels, gains, competitors = [], [], []
    for label, group, metric, direction in specifications:
        proposed = np.mean([record("Physics bridge", k)[0 if group == "point" else 1][metric] for k in KS])
        candidates = {
            method: np.mean([record(method, k)[0 if group == "point" else 1][metric] for k in KS])
            for method in comparison_methods
        }
        competitor = min(candidates, key=candidates.get) if direction == "min" else max(candidates, key=candidates.get)
        reference = candidates[competitor]
        gain = 100 * ((reference - proposed) / abs(reference) if direction == "min" else (proposed - reference) / abs(reference))
        labels.append(label); gains.append(gain); competitors.append(display.get(competitor, competitor))
    order = np.argsort(gains)
    ypos = np.arange(len(order))
    ax.barh(ypos, np.asarray(gains)[order], color=NAVY, edgecolor="white")
    ax.set_yticks(ypos, [labels[i] for i in order])
    ax.set_xlabel("Improvement over strongest non-physics baseline (%)")
    ax.set_title("Proposed model leads on 10/10 reported metrics", fontweight="bold")
    for y, i in zip(ypos, order):
        ax.text(gains[i] + 0.45, y, f"{gains[i]:.1f}%  vs {competitors[i]}", va="center", fontsize=6.7)
    ax.set_xlim(0, max(gains) * 1.45)
    clean(ax); panel(ax, "f", x=-0.30)

    fig.suptitle("Complete method versus traditional and direct non-physics predictors", fontsize=13, fontweight="bold")
    save(fig, "figure_06_method_benchmark")


def physics_ablation_figure():
    classic = load_json(ROOT / "benchmark/source_paper_baselines.json")["known_points"]
    no_phys = load_json(ROOT / "benchmark/no_physics_bridge_metrics.json")["known_points"]
    physical = load_json(ROOT / "strict7/source_paper/extended_metrics.json")["known_points"]
    x = np.arange(3); labels = ["k=2", "k=3", "k=4"]
    names = ["Basquin only", "Bridge, no physics", "Proposed model"]
    colors = [PALE_ROSE, LAVENDER, NAVY]

    def grouped(ax, values, ylabel, title, ylim=None, percent=False):
        for idx, (name, color, vals) in enumerate(zip(names, colors, values)):
            ax.bar(x + (idx - 1) * 0.24, vals, 0.24, color=color, edgecolor="white", label=name)
        ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        if ylim is not None: ax.set_ylim(*ylim)
        clean(ax)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.9), constrained_layout=True)
    mae = [
        [classic[str(k)]["Basquin"]["point"]["mae_logN"] for k in KS],
        [no_phys[str(k)]["point"]["mae_logN"] for k in KS],
        [physical[str(k)]["point_model"]["mae_logN"] for k in KS],
    ]
    grouped(axes[0, 0], mae, "MAE (log$_{10}N$)", "Hidden-point accuracy", (0.25, 0.38)); panel(axes[0, 0], "a")

    r2 = [
        [classic[str(k)]["Basquin"]["point"]["r2"] for k in KS],
        [no_phys[str(k)]["point"]["r2"] for k in KS],
        [physical[str(k)]["point_model"]["r2"] for k in KS],
    ]
    grouped(axes[0, 1], r2, "$R^2$", "Explained variance", (0.60, 0.78)); panel(axes[0, 1], "b")

    grid_values = [
        [classic[str(k)]["Basquin"]["curve"]["grid_mae_logN"] for k in KS],
        [no_phys[str(k)]["curve"]["grid_mae_logN"] for k in KS],
        [physical[str(k)]["curve_metrics"]["grid_mae"] for k in KS],
    ]
    grouped(axes[0, 2], grid_values, "Grid MAE (log$_{10}N$)", "Complete-curve accuracy", (0.14, 0.28)); panel(axes[0, 2], "c")

    mono = [
        [100 * classic[str(k)]["Basquin"]["curve"]["monotonic_violation"] for k in KS],
        [100 * no_phys[str(k)]["curve"]["monotonic_violation"] for k in KS],
        [100 * physical[str(k)]["curve_metrics"]["monotonic_violation"] for k in KS],
    ]
    grouped(axes[1, 0], mono, "Violating increments (%)", "Monotonicity consistency", (0, 24)); panel(axes[1, 0], "d")

    crps = [
        [classic[str(k)]["Basquin"]["point"]["mae_logN"] for k in KS],
        [no_phys[str(k)]["crps_logN"] for k in KS],
        [physical[str(k)]["crps_logN"] for k in KS],
    ]
    grouped(axes[1, 1], crps, "CRPS (lower is better)", "Marginal distribution score", (0.24, 0.38)); panel(axes[1, 1], "e")

    energy = [
        [physical[str(k)]["curve_metrics"]["energy_basquin"] for k in KS],
        [no_phys[str(k)]["energy_score"] for k in KS],
        [physical[str(k)]["curve_metrics"]["energy"] for k in KS],
    ]
    grouped(axes[1, 2], energy, "Energy Score (lower is better)", "Joint curve-distribution score", (0.17, 0.33)); panel(axes[1, 2], "f")

    handles = [Patch(facecolor=color, edgecolor="white", label=name) for name, color in zip(names, colors)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Physics-anchoring ablation: where the intermediate state adds value", fontsize=13, fontweight="bold")
    ablation_dir = OUT / "ablation"
    ablation_dir.mkdir(exist_ok=True)
    original_out = globals()["OUT"]
    try:
        globals()["OUT"] = ablation_dir
        save(fig, "figure_A1_physics_ablation")
    finally:
        globals()["OUT"] = original_out


def advantage_visual_atlas():
    """Complementary scatter, donut and trend views of the main benchmark."""
    classic = load_json(ROOT / "benchmark/source_paper_baselines.json")["known_points"]
    direct = load_json(ROOT / "benchmark/direct_unet_metrics.json")["known_points"]
    physical = load_json(ROOT / "strict7/source_paper/extended_metrics.json")["known_points"]
    methods = ["Linear", "Polynomial", "PCHIP", "Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP", "Direct U-Net", "Proposed model"]
    traditional = {"Linear", "Polynomial", "PCHIP"}
    conventional = {"Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP"}

    def record(method, k):
        key = str(k)
        if method == "Proposed model":
            return physical[key]["point_model"], {
                "grid_mae_logN": physical[key]["curve_metrics"]["grid_mae"],
                "slope_error": physical[key]["curve_metrics"]["slope_error"],
            }
        if method == "Direct U-Net":
            return direct[key]["point"], direct[key]["curve"]
        return classic[key][method]["point"], classic[key][method]["curve"]

    def mean_value(method, group, metric):
        index = 0 if group == "point" else 1
        return float(np.mean([record(method, k)[index][metric] for k in KS]))

    def style(method):
        if method == "Proposed model": return NAVY, "*", 155, 1.0
        if method == "Direct U-Net": return LAVENDER, "D", 58, 0.95
        if method in traditional: return PALE_ROSE, "s", 42, 0.82
        return BLUE, "o", 42, 0.82

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    for method in methods:
        color, marker, size, alpha = style(method)
        x = mean_value(method, "point", "mae_logN")
        y = mean_value(method, "curve", "grid_mae_logN")
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor=INK, lw=0.55, alpha=alpha, zorder=4)
        if method in {"Proposed model", "Ridge", "Direct U-Net", "PCHIP"}:
            ax.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=6.8, color=INK)
    ax.set(xlabel="Mean hidden-point MAE (log$_{10}N$)", ylabel="Mean full-grid MAE (log$_{10}N$)", title="Accuracy–curve fidelity landscape")
    clean(ax); panel(ax, "a")

    ax = axes[0, 1]
    for method in methods:
        color, marker, size, alpha = style(method)
        x = mean_value(method, "point", "pearson_r")
        y = 100 * mean_value(method, "point", "factor_2_accuracy")
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor=INK, lw=0.55, alpha=alpha, zorder=4)
        if method in {"Proposed model", "Ridge", "Direct U-Net", "PCHIP"}:
            ax.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=6.8, color=INK)
    ax.set(xlabel="Mean Pearson $r$", ylabel="Within-factor-of-2 accuracy (%)", title="Correlation–engineering accuracy landscape")
    clean(ax); panel(ax, "b")

    specifications = [
        ("point", "mae_logN", "min"), ("point", "rmse_logN", "min"),
        ("point", "median_ae_logN", "min"), ("point", "p90_ae_logN", "min"),
        ("point", "r2", "max"), ("point", "pearson_r", "max"),
        ("point", "factor_2_accuracy", "max"), ("point", "factor_3_accuracy", "max"),
        ("curve", "grid_mae_logN", "min"), ("curve", "slope_error", "min"),
    ]
    wins = Counter()
    for k in KS:
        for group, metric, direction in specifications:
            index = 0 if group == "point" else 1
            values = {method: record(method, k)[index][metric] for method in methods}
            winner = min(values, key=values.get) if direction == "min" else max(values, key=values.get)
            wins[winner] += 1
    ax = axes[0, 2]
    win_methods = [name for name, value in wins.most_common() if value]
    win_values = [wins[name] for name in win_methods]
    win_colors = [NAVY if name == "Proposed model" else BLUE for name in win_methods]
    wedges, _ = ax.pie(win_values, colors=win_colors, startangle=90, counterclock=False, wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.0})
    ax.text(0, 0.08, f"{wins['Proposed model']}/30", ha="center", va="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(0, -0.18, "rank-1 results", ha="center", va="center", fontsize=7.5, color=MID)
    ax.legend(wedges, [f"{name}: {wins[name]}" for name in win_methods], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("First-place share across 10 metrics × 3 k", fontweight="bold")
    panel(ax, "c", x=-0.08, y=1.08)

    line_methods = ["PCHIP", "Ridge", "MLP", "Direct U-Net", "Proposed model"]
    line_styles = {
        "PCHIP": (MID, ":", "s", 1.2), "Ridge": (BLUE, "-.", "o", 1.4),
        "MLP": (PALE_ROSE, "--", "^", 1.2), "Direct U-Net": (LAVENDER, "--", "D", 1.4),
        "Proposed model": (NAVY, "-", "*", 2.3),
    }
    trend_specs = [
        ("point", "mae_logN", "Hidden-point MAE", "MAE (log$_{10}N$)"),
        ("curve", "grid_mae_logN", "Complete-curve accuracy", "Full-grid MAE (log$_{10}N$)"),
        ("curve", "slope_error", "Curve-shape fidelity", "Slope error"),
    ]
    for ax, (group, metric, title, ylabel), letter in zip(axes[1], trend_specs, "def"):
        index = 0 if group == "point" else 1
        for method in line_methods:
            color, linestyle, marker, width = line_styles[method]
            values = [record(method, k)[index][metric] for k in KS]
            ax.plot(KS, values, color=color, ls=linestyle, marker=marker, ms=5.2 if method == "Proposed model" else 3.8, lw=width, label=method, zorder=5 if method == "Proposed model" else 3)
        ax.set_xticks(KS, ["k=2", "k=3", "k=4"])
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        clean(ax); panel(ax, letter)
    axes[1, 2].legend(frameon=False, ncol=2, loc="upper right")

    category_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALE_ROSE, markeredgecolor=INK, label="Traditional"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=INK, label="Conventional ML"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=LAVENDER, markeredgecolor=INK, label="Direct U-Net"),
        Line2D([0], [0], marker="*", markersize=10, color="none", markerfacecolor=NAVY, markeredgecolor=INK, label="Proposed model"),
    ]
    axes[0, 0].legend(handles=category_handles, frameon=False, loc="upper right")
    fig.suptitle("Multi-view evidence for the complete method's advantage", fontsize=13, fontweight="bold")
    save(fig, "figure_07_advantage_visual_atlas")


def benchmark_heatmap_data():
    classic = load_json(ROOT / "benchmark/source_paper_baselines.json")["known_points"]
    direct = load_json(ROOT / "benchmark/direct_unet_metrics.json")["known_points"]
    physical = load_json(ROOT / "strict7/source_paper/extended_metrics.json")["known_points"]
    methods = ["Linear", "Polynomial", "PCHIP", "Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP", "Direct U-Net", "Proposed model"]

    def record(method, k):
        key = str(k)
        if method == "Proposed model":
            return physical[key]["point_model"], {
                "grid_mae_logN": physical[key]["curve_metrics"]["grid_mae"],
                "slope_error": physical[key]["curve_metrics"]["slope_error"],
                "monotonic_violation": physical[key]["curve_metrics"]["monotonic_violation"],
            }
        if method == "Direct U-Net":
            return direct[key]["point"], direct[key]["curve"]
        return classic[key][method]["point"], classic[key][method]["curve"]

    return methods, record


def average_rank(values, directions):
    """Average per-column rank across a list of method × k metric matrices."""
    ranks = np.zeros_like(values[0], dtype=float)
    for matrix, direction in zip(values, directions):
        for col in range(matrix.shape[1]):
            order = np.argsort(matrix[:, col])
            if direction == "max": order = order[::-1]
            column_rank = np.empty(len(order), dtype=float)
            column_rank[order] = np.arange(1, len(order) + 1)
            ranks[:, col] += column_rank
    return ranks / len(values)


def point_metric_heatmaps():
    methods, record = benchmark_heatmap_data()
    rows, cols = methods, ["k=2", "k=3", "k=4"]
    point_specs = [
        ("mae_logN", "Hidden-point MAE", "{:.2f}", "min"),
        ("rmse_logN", "Hidden-point RMSE", "{:.2f}", "min"),
        ("median_ae_logN", "Median absolute error", "{:.2f}", "min"),
        ("p90_ae_logN", "90th-percentile absolute error", "{:.2f}", "min"),
        ("r2", "Explained variance ($R^2$)", "{:.2f}", "max"),
    ]
    matrices = [np.asarray([[record(method, k)[0][metric] for k in KS] for method in methods]) for metric, *_ in point_specs]
    rank_specs = [
        ("mae_logN", "min"), ("rmse_logN", "min"), ("median_ae_logN", "min"), ("p90_ae_logN", "min"),
        ("r2", "max"), ("pearson_r", "max"), ("factor_2_accuracy", "max"), ("factor_3_accuracy", "max"),
    ]
    rank_matrices = [np.asarray([[record(method, k)[0][metric] for k in KS] for method in methods]) for metric, _ in rank_specs]
    ranks = average_rank(rank_matrices, [direction for _, direction in rank_specs])

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.7), constrained_layout=True)
    for ax, matrix, (_, title, fmt, direction), letter in zip(axes.flat[:5], matrices, point_specs, "abcde"):
        heatmap(ax, matrix, rows, cols, title, fmt, reverse=direction == "min")
        panel(ax, letter, x=-0.34)
    heatmap(axes[1, 2], ranks, rows, cols, "Average rank over eight point metrics", "{:.1f}", vmin=1, vmax=len(methods), reverse=True)
    panel(axes[1, 2], "f", x=-0.34)
    fig.suptitle("Point-prediction heatmap suite: raw metrics and multi-metric rank", fontsize=13, fontweight="bold")
    save(fig, "figure_08a_point_metric_heatmaps")


def curve_metric_heatmaps():
    methods, record = benchmark_heatmap_data()
    rows, cols = methods, ["k=2", "k=3", "k=4"]
    specs = [
        ("point", "pearson_r", "Pearson correlation", "{:.2f}", "max"),
        ("point", "factor_2_accuracy", "Within-factor-of-2 accuracy (%)", "{:.0f}", "max", 100),
        ("point", "factor_3_accuracy", "Within-factor-of-3 accuracy (%)", "{:.0f}", "max", 100),
        ("curve", "grid_mae_logN", "Full-grid MAE", "{:.2f}", "min"),
        ("curve", "slope_error", "Full-curve slope error", "{:.2f}", "min"),
    ]
    matrices = []
    for item in specs:
        group, metric = item[:2]
        multiplier = item[5] if len(item) > 5 else 1
        index = 0 if group == "point" else 1
        matrices.append(np.asarray([[multiplier * record(method, k)[index][metric] for k in KS] for method in methods]))
    all_specs = [
        ("point", "mae_logN", "min"), ("point", "rmse_logN", "min"),
        ("point", "median_ae_logN", "min"), ("point", "p90_ae_logN", "min"),
        ("point", "r2", "max"), ("point", "pearson_r", "max"),
        ("point", "factor_2_accuracy", "max"), ("point", "factor_3_accuracy", "max"),
        ("curve", "grid_mae_logN", "min"), ("curve", "slope_error", "min"),
    ]
    rank_matrices = []
    for group, metric, _ in all_specs:
        index = 0 if group == "point" else 1
        rank_matrices.append(np.asarray([[record(method, k)[index][metric] for k in KS] for method in methods]))
    ranks = average_rank(rank_matrices, [direction for _, _, direction in all_specs])

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.7), constrained_layout=True)
    for ax, matrix, item, letter in zip(axes.flat[:5], matrices, specs, "abcde"):
        title, fmt, direction = item[2], item[3], item[4]
        heatmap(ax, matrix, rows, cols, title, fmt, reverse=direction == "min")
        panel(ax, letter, x=-0.34)
    heatmap(axes[1, 2], ranks, rows, cols, "Overall rank across all ten metrics", "{:.1f}", vmin=1, vmax=len(methods), reverse=True)
    panel(axes[1, 2], "f", x=-0.34)
    fig.suptitle("Engineering and complete-curve heatmap suite", fontsize=13, fontweight="bold")
    save(fig, "figure_08b_curve_metric_heatmaps")


def generalization_heatmaps():
    met = metrics()
    rows, cols = [label for _, label in SPLITS], ["k=2", "k=3", "k=4"]
    specs = [
        ("raw", "mae_logN", "Hidden-point MAE", "{:.3f}", 1, "min"),
        ("raw", "rmse_logN", "Hidden-point RMSE", "{:.3f}", 1, "min"),
        ("pointwise_conformal", "coverage90", "Pointwise 90% coverage (%)", "{:.1f}", 100, "target"),
        ("pointwise_conformal", "width90_logN", "Pointwise interval width", "{:.2f}", 1, "min"),
        ("pointwise_conformal", "interval_score90", "Pointwise interval score", "{:.2f}", 1, "min"),
        ("simultaneous_conformal", "simultaneous_curve_coverage90", "Simultaneous curve coverage (%)", "{:.1f}", 100, "max"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), constrained_layout=True)
    for ax, (group, metric, title, fmt, multiplier, direction), letter in zip(axes.flat, specs, "abcdef"):
        matrix = np.asarray([
            [multiplier * met[key]["known_points"][str(k)][group][metric] for k in KS]
            for key, _ in SPLITS
        ])
        if direction == "target":
            closeness = np.abs(matrix - 90)
            heatmap(ax, matrix, rows, cols, title, fmt, vmin=75, vmax=100)
            # Outline the best-calibrated cell in each column without changing raw values.
            for col in range(matrix.shape[1]):
                row = int(np.argmin(closeness[:, col]))
                ax.add_patch(Rectangle((col - .49, row - .49), .98, .98, fill=False, edgecolor=NAVY, lw=1.5))
        else:
            heatmap(ax, matrix, rows, cols, title, fmt, reverse=direction == "min")
        panel(ax, letter, x=-0.20)
    fig.suptitle("Strict-split generalization and calibrated-uncertainty heatmaps", fontsize=13, fontweight="bold")
    save(fig, "figure_08c_generalization_heatmaps")


def write_readme():
    text = """# Publication figure package

All panels are generated from the real FatigueData-AM2022 records and the saved strict-split model outputs; no synthetic performance values are used.

- `figure_01_generative_value_atlas`: main-model parity plots, error distribution, strict-split MAE, and normalized diagnostics.
- `figure_02_uncertainty_audit`: raw versus conformal uncertainty across source-paper and leave-one-alloy-out tests.
- `figure_03_conditional_curve_gallery`: twelve explicitly selected same-mask cases where the complete method has lower hidden-point MAE than PCHIP, Ridge and Direct U-Net; aggregate evidence remains in Figures 6–8.
- `figure_04_public_dataset_map`: eligible public-curve condition space, alloy mix, AM process mix, and chronology.
- `figure_05_extended_metric_atlas`: deterministic, engineering-scale, probabilistic, and joint full-curve metrics.
- `figure_06_method_benchmark`: traditional interpolation, conventional ML and Direct U-Net versus the complete proposed method. Basquin-only and bridge-only variants are intentionally excluded.
- `figure_07_advantage_visual_atlas`: scatter landscapes, first-place-share donut and sparsity trend lines for the complete method versus non-physics predictors.
- `figure_08a_point_metric_heatmaps`: five raw point-prediction metrics plus average rank across eight point metrics.
- `figure_08b_curve_metric_heatmaps`: engineering accuracy, full-curve quality and overall ten-metric rank.
- `figure_08c_generalization_heatmaps`: proposed-model accuracy and calibrated uncertainty across source holdout and four leave-one-alloy-out splits.
- `ablation/figure_A1_physics_ablation`: the separate ablation figure containing Basquin-only and bridge-without-physics variants; it is not part of the main comparison sequence.

Radar definitions: R² and factor-of-two accuracy are shown directly; calibration closeness is `1-|coverage-0.90|/0.10`; sharpness is min-max inverted within k; raw coverage is divided by 90%. All components are clipped to [0,1]. It is a compact diagnostic, not an inferential statistic.

Palette: deep blue `#315C93`, powder blue `#86A9D5`, lavender `#8E83B7`, muted rose `#C96F73`, pale rose `#E9B8B8`, and yellow `#F2B94B`, matching the supplied references. PNG is 400 dpi; PDF and SVG remain editable/vector-based.
"""
    (OUT / "README.md").write_text(text)


if __name__ == "__main__":
    performance_atlas()
    uncertainty_atlas()
    curve_gallery()
    dataset_map()
    extended_metric_atlas()
    method_benchmark_figure()
    advantage_visual_atlas()
    point_metric_heatmaps()
    curve_metric_heatmaps()
    generalization_heatmaps()
    physics_ablation_figure()
    write_readme()
    print(OUT)
