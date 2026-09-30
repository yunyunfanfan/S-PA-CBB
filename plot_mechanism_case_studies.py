#!/usr/bin/env python3
"""Mechanism case studies and a selective-use map for S-PA-CBB.

The case gallery is intentionally explanatory: cases are screened after the
aggregate evaluation and never contribute to the headline benchmark.  The
selective-use map includes every held-out curve under four deterministic sparse
layouts, so it remains a population-level audit rather than a case montage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import types
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def install_npu_checkpoint_compatibility():
    names = (
        "torch_npu", "torch_npu.utils", "torch_npu.utils.storage",
        "torch_npu.npu", "torch_npu.npu.random", "torch_npu.npu._format",
    )
    modules = {}
    for name in names:
        module = types.ModuleType(name); module.__path__ = []
        modules[name] = module; sys.modules.setdefault(name, module)

    def rebuild(*args):
        return torch._utils._rebuild_tensor_v2(*args[:6])

    class Format:
        def __init__(self, *args, **kwargs):
            pass

    modules["torch_npu.utils.storage"]._rebuild_npu_tensor = rebuild
    modules["torch_npu.npu._format"].Format = Format


install_npu_checkpoint_compatibility()

from ablation_control_variants import sample  # noqa: E402
from control_bridge import BaseUNet, ControlBridge, make_case  # noqa: E402
from control_score_bridge import split_curves  # noqa: E402
from downstream_fatigue_strength import invert_log_stress, monotone_decreasing  # noqa: E402


NAVY = "#315F98"
BLUE = "#84A9D4"
LIGHT_BLUE = "#DDE8F4"
ROSE = "#C96F72"
LIGHT_ROSE = "#F0D3D3"
GOLD = "#E9AE3A"
PURPLE = "#8A7FB8"
INK = "#27323A"
GREY = "#7B858E"
PALETTE = [NAVY, BLUE, PURPLE, ROSE, GOLD, GREY]


def stable_seed(*parts):
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31)


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8.0,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def choose_obs(curve, layout, mask_seed):
    order = np.argsort(curve["x"])
    if layout == "low_stress":
        return np.sort(order[:2])
    if layout == "high_stress":
        return np.sort(order[-2:])
    if layout == "endpoints":
        return np.sort(np.asarray([order[0], order[-1]]))
    rng = np.random.default_rng(stable_seed(curve["id"], 2, mask_seed))
    return np.sort(rng.choice(len(curve["x"]), 2, replace=False))


def curve_geometry(curve):
    x, y = curve["gx"].astype(float), curve["gy"].astype(float)
    linear = np.polyval(np.polyfit(x, y, 1), x)
    linear_rmse = float(np.sqrt(np.mean((y - linear) ** 2)))
    best_rmse, best_split, best_slopes = linear_rmse, len(x) // 2, (0.0, 0.0)
    for split in range(5, len(x) - 5):
        left = np.polyfit(x[:split + 1], y[:split + 1], 1)
        right = np.polyfit(x[split:], y[split:], 1)
        pred = np.r_[np.polyval(left, x[:split]), np.polyval(right, x[split:])]
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_split, best_slopes = rmse, split, (left[0], right[0])
    piecewise_gain = max(0.0, 1.0 - best_rmse / max(linear_rmse, 1e-8))
    slope_change = float(abs(best_slopes[0] - best_slopes[1]))
    if linear_rmse < 0.055:
        curve_type = "single slope"
    elif piecewise_gain > 0.35 and slope_change > 1.0:
        curve_type = "knee / multi-slope"
    else:
        curve_type = "smooth curvature"
    return {"curvature_rmse": linear_rmse, "piecewise_gain": piecewise_gain,
            "slope_change": slope_change, "knee_index": int(best_split),
            "curve_type": curve_type}


def strength_samples(gx, curves, target_life=6.0):
    values = []
    for curve in curves:
        monotone = monotone_decreasing(np.asarray(curve, float))
        if monotone.min() <= target_life <= monotone.max():
            values.append(float(10 ** invert_log_stress(gx, monotone, target_life)))
        else:
            values.append(float("nan"))
    return np.asarray(values)


def group_family(family):
    text = str(family).lower()
    if "ti" in text:
        return "Ti alloys"
    if "in718" in text or "nickel" in text:
        return "Ni alloys"
    if "316" in text or "steel" in text:
        return "Steels"
    if "alsi" in text or text.startswith("al"):
        return "Al alloys"
    return "Other"


def evaluate(curves, model, checkpoint, device, layouts, ensemble, steps, eta,
             mask_seed, batch_size):
    records = []
    for layout in layouts:
        prepared = []
        for curve in curves:
            obs = choose_obs(curve, layout, mask_seed)
            case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
            prepared.append((curve, case, curve_geometry(curve)))
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            tensors = []
            for name in ("static", "control", "raw", "phys", "mask", "oy"):
                value = torch.tensor(np.stack([case[name] for _, case, _ in batch]), device=device)
                value = value[:, None].repeat(1, ensemble, *([1] * (value.ndim - 1)))
                tensors.append(value.reshape(len(batch) * ensemble, *value.shape[2:]))
            torch.manual_seed(stable_seed("mechanism", layout, start, mask_seed))
            with torch.no_grad():
                normalized = sample(model, "full", *tensors, steps, eta)
            draws_all = normalized.cpu().numpy().reshape(len(batch), ensemble, -1)
            draws_all = draws_all * checkpoint["ystd"] + checkpoint["ymean"]
            for (curve, case, geometry), draws in zip(batch, draws_all):
                physics = case["phys"] * checkpoint["ystd"] + checkpoint["ymean"]
                mean = draws.mean(axis=0); truth = curve["gy"]
                physics_mae = float(np.mean(np.abs(physics - truth)))
                generated_mae = float(np.mean(np.abs(mean - truth)))
                coverage = float(np.ptp(curve["x"][case["obs"]]) / max(np.ptp(curve["gx"]), 1e-8))
                spread = float(np.mean(np.quantile(draws, .95, axis=0)
                                       - np.quantile(draws, .05, axis=0)))
                correction = float(np.mean(np.abs(mean - physics)))
                truth_strength = strength_samples(curve["gx"], [truth])[0]
                physics_strength = strength_samples(curve["gx"], [physics])[0]
                draw_strength = strength_samples(curve["gx"], draws)
                finite_strength = draw_strength[np.isfinite(draw_strength)]
                generated_strength = float(finite_strength.mean()) if len(finite_strength) else float("nan")
                if math.isfinite(truth_strength) and math.isfinite(physics_strength) and math.isfinite(generated_strength):
                    strength_improvement = (abs(physics_strength - truth_strength)
                                            - abs(generated_strength - truth_strength))
                else:
                    strength_improvement = float("nan")
                records.append({
                    "curve_id": curve["id"], "family": curve["family"],
                    "family_group": group_family(curve["family"]), "layout": layout,
                    "obs": case["obs"].astype(int).tolist(), "coverage": coverage,
                    "physics_mae": physics_mae, "generated_mae": generated_mae,
                    "improvement": physics_mae - generated_mae,
                    "correction_magnitude": correction, "width90": spread,
                    "truth_strength": float(truth_strength),
                    "physics_strength": float(physics_strength),
                    "generated_strength": generated_strength,
                    "strength_improvement": strength_improvement,
                    "draw_strength": draw_strength,
                    "curve": curve, "physics": physics, "draws": draws,
                    "raw_draws": draws.copy(),
                    **geometry,
                })
    return records


def apply_selective_gate(records, gate):
    """Translate the posterior mean while preserving posterior deviations."""
    for record in records:
        raw = record["raw_draws"]
        raw_mean = raw.mean(axis=0)
        residual = raw_mean - record["physics"]
        endpoint = .5 * (abs(residual[0]) + abs(residual[-1]))
        if gate["score"] == "tail-aware residual":
            score = (np.mean(np.abs(residual)) + .5 * endpoint) / (record["width90"] + 1e-4)
        elif gate["score"] == "support-weighted residual":
            score = np.mean(np.abs(residual)) * math.sqrt(record["coverage"] + .05) / (record["width90"] + 1e-4)
        else:
            score = np.mean(np.abs(residual)) / (record["width90"] + 1e-4)
        active = score >= gate["threshold"]
        mean = record["physics"] + (gate["alpha"] if active else 0.0) * residual
        draws = mean[None, :] + (raw - raw_mean[None, :])
        truth = record["curve"]["gy"]
        record.update({"draws": draws, "gate_score": float(score), "gate_active": bool(active),
                       "generated_mae": float(np.mean(np.abs(mean-truth))),
                       "improvement": record["physics_mae"]-float(np.mean(np.abs(mean-truth))),
                       "correction_magnitude": float(np.mean(np.abs(mean-record["physics"])))})
        draw_strength = strength_samples(record["curve"]["gx"], draws)
        finite = draw_strength[np.isfinite(draw_strength)]
        record["draw_strength"] = draw_strength
        record["generated_strength"] = float(finite.mean()) if len(finite) else float("nan")
        if math.isfinite(record["truth_strength"]) and math.isfinite(record["physics_strength"]) and len(finite):
            record["strength_improvement"] = (abs(record["physics_strength"]-record["truth_strength"])
                                               - abs(record["generated_strength"]-record["truth_strength"]))
    return records


def eligible_strength(record):
    values = record["draw_strength"]
    return (math.isfinite(record["truth_strength"]) and math.isfinite(record["physics_strength"])
            and np.mean(np.isfinite(values)) >= 0.8)


def select_cases(records):
    selected = []
    used = set()

    def pick(candidates, key, reverse=False):
        candidates = [r for r in candidates if r["curve_id"] not in used and eligible_strength(r)]
        choice = sorted(candidates, key=key, reverse=reverse)[0]
        selected.append(choice); used.add(choice["curve_id"]); return choice

    # Easy physics: broad coverage, almost linear truth, and a small learned correction.
    pick([r for r in records if r["layout"] == "endpoints" and r["curvature_rmse"] < 0.055
          and r["physics_mae"] < 0.08 and r["generated_mae"] < 0.10],
         key=lambda r: r["correction_magnitude"] + abs(r["improvement"]))

    # Curved/knee response: broad-enough observations and the largest verified gain.
    pick([r for r in records if r["layout"] in ("random", "endpoints")
          and r["coverage"] > 0.55 and r["curvature_rmse"] > 0.10],
         key=lambda r: r["improvement"], reverse=True)

    # One-sided extrapolation in both directions, again requiring a positive gain.
    def decision_score(r):
        curve_gain = r["improvement"] / max(r["physics_mae"], 1e-8)
        base_strength_error = abs(r["physics_strength"] - r["truth_strength"])
        strength_gain = r["strength_improvement"] / max(base_strength_error, 1e-8)
        return curve_gain + strength_gain

    pick([r for r in records if r["layout"] == "high_stress" and r["improvement"] > 0
          and r["strength_improvement"] > 0], key=decision_score, reverse=True)
    pick([r for r in records if r["layout"] == "low_stress" and r["improvement"] > 0
          and r["strength_improvement"] > 0], key=decision_score, reverse=True)
    return selected


def save_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_case_gallery(selected, stem):
    style(); fig, axes = plt.subplots(4, 4, figsize=(7.35, 8.05))
    titles = ("Sparse tests + physics", "Residual correction",
              "Selective ensemble ($n=32$)", r"$10^6$-cycle strength")
    row_names = ("Physics sufficient", "Curvature / knee", "High-stress anchors",
                 "Low-stress anchors")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, weight="bold", pad=7, fontsize=7.3)

    for row, record in enumerate(selected):
        curve, physics, draws = record["curve"], record["physics"], record["draws"]
        gx, truth = curve["gx"], curve["gy"]; stress = 10 ** gx
        mean = draws.mean(axis=0); low, high = np.quantile(draws, [.05, .95], axis=0)
        obs = np.asarray(record["obs"], int)

        ax = axes[row, 0]
        ax.plot(stress, truth, color=INK, lw=1.55, label="Complete truth")
        ax.plot(stress, physics, color=ROSE, lw=1.5, ls="--", label="Basquin state")
        ax.scatter(10 ** curve["x"][obs], curve["y"][obs], s=25, color=GOLD,
                   edgecolor=INK, linewidth=.45, zorder=5, label="Sparse tests")
        ax.text(.03, .05, f"{row_names[row]}\n{curve['id'].replace('FatigueData-AM2022:', 'AM2022:')} | {curve['family']}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, weight="bold",
                bbox=dict(boxstyle="round,pad=.22", fc="white", ec=LIGHT_BLUE, alpha=.92))
        ax.text(.97, .96, f"coverage = {record['coverage']:.2f}\nBasquin MAE = {record['physics_mae']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.1)

        ax = axes[row, 1]
        required = truth - physics; learned = mean - physics
        ax.axhline(0, color=GREY, lw=.7)
        ax.fill_between(stress, low - physics, high - physics, color=BLUE, alpha=.28,
                        linewidth=0, label="Generated 90% residual")
        ax.plot(stress, required, color=INK, lw=1.45, label="Required residual")
        ax.plot(stress, learned, color=NAVY, lw=1.45, label="Learned mean residual")
        ax.text(.97, .96, f"gate = {'on' if record.get('gate_active') else 'off'}\n"
                f"MAE gain = {record['improvement']:+.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1)

        ax = axes[row, 2]
        for draw in draws[:12]:
            ax.plot(stress, draw, color=BLUE, alpha=.14, lw=.5)
        ax.fill_between(stress, low, high, color=BLUE, alpha=.30, linewidth=0,
                        label="Generated 90% interval")
        ax.plot(stress, mean, color=NAVY, lw=1.5, label="Selective mean")
        ax.plot(stress, truth, color=INK, lw=1.45, ls="--", label="Complete truth")
        ax.scatter(10 ** curve["x"][obs], curve["y"][obs], s=19, color=GOLD,
                   edgecolor=INK, linewidth=.4, zorder=5)
        ax.text(.97, .96, f"S-PA-CBB MAE = {record['generated_mae']:.3f}\n"
                f"raw width = {record['width90']:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1)

        ax = axes[row, 3]
        strengths = record["draw_strength"]; strengths = strengths[np.isfinite(strengths)]
        lo_s, hi_s = np.quantile(strengths, [.05, .95]); mean_s = strengths.mean()
        span = max(np.ptp(strengths), abs(record["physics_strength"] - record["truth_strength"]), 1.0)
        bins = np.linspace(min(strengths.min(), record["physics_strength"], record["truth_strength"]) - .08 * span,
                           max(strengths.max(), record["physics_strength"], record["truth_strength"]) + .08 * span, 11)
        ax.hist(strengths, bins=bins, color=LIGHT_BLUE, edgecolor="white", density=True)
        ax.axvspan(lo_s, hi_s, color=BLUE, alpha=.22, lw=0)
        ax.axvline(record["truth_strength"], color=INK, lw=1.6, label="Truth")
        ax.axvline(record["physics_strength"], color=ROSE, lw=1.4, ls="--", label="Basquin")
        ax.axvline(mean_s, color=NAVY, lw=1.5, label="Generated mean")
        base_error = abs(record["physics_strength"] - record["truth_strength"])
        gen_error = abs(mean_s - record["truth_strength"])
        ax.text(.97, .95, f"Basquin error = {base_error:.1f} MPa\n"
                f"generated error = {gen_error:.1f} MPa", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1)
        ax.set_yticks([])

        for col in range(4):
            ax = axes[row, col]
            ax.grid(True, ls="--", lw=.4, alpha=.30)
            ax.tick_params(direction="in", length=2.5)
            if col < 3:
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            if row < 3:
                ax.set_xticklabels([])
        axes[row, 0].set_ylabel(r"$\log_{10}N$")
        axes[row, 1].set_ylabel(r"Residual $\Delta\log_{10}N$")
        axes[row, 2].set_yticklabels([])

    for col in range(3):
        axes[-1, col].set_xlabel("Stress amplitude (MPa)")
    axes[-1, 3].set_xlabel(r"$\sigma_{10^6}$ (MPa)")

    # Compact column-specific legends avoid repeating nine entries in every row.
    handles0, labels0 = axes[0, 0].get_legend_handles_labels()
    handles1, labels1 = axes[0, 1].get_legend_handles_labels()
    handles2, labels2 = axes[0, 2].get_legend_handles_labels()
    handles3, labels3 = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles0 + handles1 + handles2[1:2] + handles3,
               labels0 + labels1 + labels2[1:2] + labels3,
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .008),
               handlelength=2.1, columnspacing=1.15)
    fig.suptitle("Why the physical state fails - and what conditional generation changes",
                 fontsize=11.2, weight="bold", y=.992)
    fig.subplots_adjust(left=.085, right=.988, top=.925, bottom=.145, wspace=.27, hspace=.14)
    save_figure(fig, stem)


def binned_means(records, x_edges, y_edges):
    sums = np.zeros((len(y_edges) - 1, len(x_edges) - 1)); counts = np.zeros_like(sums)
    for r in records:
        ix = np.searchsorted(x_edges, r["physics_mae"], side="right") - 1
        iy = np.searchsorted(y_edges, r["coverage"], side="right") - 1
        if 0 <= ix < sums.shape[1] and 0 <= iy < sums.shape[0]:
            sums[iy, ix] += r["improvement"]; counts[iy, ix] += 1
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return means, counts


def plot_selective_map(records, selected, stem):
    style()
    fig = plt.figure(figsize=(7.35, 3.70))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.17), hspace=.24, wspace=.30)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[1, 0]); legend_ax.axis("off")
    rule_ax = fig.add_subplot(grid[1, 1]); rule_ax.axis("off")
    x = np.asarray([r["physics_mae"] for r in records]); y = np.asarray([r["coverage"] for r in records])
    gain = np.asarray([r["improvement"] for r in records])
    limit = float(np.quantile(np.abs(gain), .96)); norm = mpl.colors.TwoSlopeNorm(0, -limit, limit)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("gain", [ROSE, "#F8F7F4", NAVY])
    markers = {"single slope": "o", "smooth curvature": "s", "knee / multi-slope": "^"}

    ax = axes[0]
    for curve_type, marker in markers.items():
        ids = np.asarray([r["curve_type"] == curve_type for r in records])
        ax.scatter(x[ids], y[ids], c=gain[ids], cmap=cmap, norm=norm, marker=marker,
                   s=19, alpha=.62, edgecolors=INK, linewidths=.18, label=curve_type)
    for index, r in enumerate(selected, 1):
        ax.scatter(r["physics_mae"], r["coverage"], marker="*", s=92,
                   facecolor=GOLD, edgecolor=INK, linewidth=.65, zorder=8)
        ax.annotate(str(index), (r["physics_mae"], r["coverage"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=6.5, weight="bold")
    ax.set_xscale("log"); ax.set_xlim(max(x.min() * .75, .008), x.max() * 1.25); ax.set_ylim(-.04, 1.04)
    ax.set_xlabel("Sparse Basquin grid MAE (log cycles)")
    ax.set_ylabel("Observed stress-span coverage")
    ax.set_title("Held-out curve-mask cases", weight="bold", pad=6)
    ax.axvline(.25, color=INK, ls=(0, (3, 2)), lw=.8); ax.axhline(.55, color=INK, ls=(0, (3, 2)), lw=.8)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=3, frameon=False,
                     fontsize=5.8, handletextpad=.30, columnspacing=.65)

    ax = axes[1]
    x_edges = np.geomspace(max(x.min() * .85, .008), x.max() * 1.05, 8)
    y_edges = np.linspace(0, 1, 7)
    means, counts = binned_means(records, x_edges, y_edges)
    mesh = ax.pcolormesh(x_edges, y_edges, means, cmap=cmap, norm=norm, shading="flat")
    for iy in range(means.shape[0]):
        for ix in range(means.shape[1]):
            if counts[iy, ix] >= 3 and np.isfinite(means[iy, ix]):
                center_x = math.sqrt(x_edges[ix] * x_edges[ix + 1])
                center_y = (y_edges[iy] + y_edges[iy + 1]) / 2
                ax.text(center_x, center_y, f"{means[iy, ix]:+.2f}", ha="center", va="center",
                        fontsize=5.7, color=INK)
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("Sparse Basquin grid MAE (log cycles)")
    ax.set_ylabel("Observed stress-span coverage")
    ax.set_title("Binned selective-correction benefit", weight="bold", pad=6)
    ax.axvline(.25, color=INK, ls=(0, (3, 2)), lw=.9); ax.axhline(.55, color=INK, ls=(0, (3, 2)), lw=.9)
    # Keep the decision rule outside the heatmap.  Four compact two-line cells
    # remain legible at single-column manuscript scale and cannot cover values.
    rule_ax.text(.02, .72, "Broad + low mismatch\nretain physics",
                 fontsize=5.2, weight="bold", ha="left", va="center", linespacing=1.0)
    rule_ax.text(.02, .22, "Broad + high mismatch\ntransform residual",
                 fontsize=5.2, weight="bold", ha="left", va="center", linespacing=1.0)
    rule_ax.text(.54, .72, "Narrow support\nadd stress test",
                 fontsize=5.2, weight="bold", ha="left", va="center", linespacing=1.0)
    rule_ax.text(.54, .22, "Extreme shift\nabstain / recalibrate",
                 fontsize=5.2, weight="bold", ha="left", va="center", linespacing=1.0)
    colorbar = fig.colorbar(mesh, ax=axes[1], fraction=.05, pad=.04)
    colorbar.set_label("Grid-MAE reduction: Basquin - S-PA-CBB")
    for ax in axes:
        ax.grid(True, which="major", ls="--", lw=.4, alpha=.25); ax.tick_params(direction="in")
    fig.suptitle("Stage I: identify curves requiring transformation",
                 fontsize=10.4, weight="bold", y=.985)
    fig.subplots_adjust(left=.085, right=.93, top=.87, bottom=.075)
    save_figure(fig, stem)


def serializable_record(record):
    values = record["draw_strength"]; finite = values[np.isfinite(values)]
    return {key: record[key] for key in (
        "curve_id", "family", "family_group", "layout", "obs", "coverage",
        "physics_mae", "generated_mae", "improvement", "correction_magnitude",
        "width90", "truth_strength", "physics_strength", "generated_strength",
        "strength_improvement", "curvature_rmse",
        "piecewise_gain", "slope_change", "knee_index", "curve_type")
    } | {"generated_strength_mean": float(finite.mean()),
         "generated_strength_q05_q95": np.quantile(finite, [.05, .95]).tolist()}


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split_seed = checkpoint.get("split_seed", checkpoint.get("seed", 20260928))
    _, _, curves, _ = split_curves(args.data, split_seed,
                                    split_ids=checkpoint.get("split_ids"))
    if args.database_prefix:
        curves = [curve for curve in curves if curve["id"].startswith(args.database_prefix)]
    device = torch.device("cpu")
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    layouts = ("random", "endpoints", "high_stress", "low_stress")
    records = evaluate(curves, model, checkpoint, device, layouts, args.ensemble,
                       args.steps, args.eta, args.mask_seed, args.batch)
    gate_payload = json.loads(args.gate_results.read_text())
    gate = gate_payload["gates"]["2"]
    records = apply_selective_gate(records, gate)
    selected = select_cases(records)

    case_stem = args.output_dir / "mechanism_case_studies"
    map_stem = args.output_dir / "selective_use_mechanism_map"
    plot_case_gallery(selected, case_stem)
    plot_selective_map(records, selected, map_stem)

    args.paper_figures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case_stem.with_suffix(".pdf"), args.paper_figures / "mechanism_case_studies.pdf")
    shutil.copy2(map_stem.with_suffix(".pdf"), args.paper_figures / "selective_use_mechanism_map.pdf")

    manifest = {
        "status": "post-hoc explanatory screening; aggregate map uses all held-out cases",
        "checkpoint": str(args.checkpoint),
        "protocol": {"curves": len(curves), "layouts": list(layouts),
                     "cases": len(records), "steps": args.steps,
                     "ensemble": args.ensemble, "eta": args.eta},
        "selective_gate": gate,
        "case_selection": [serializable_record(r) for r in selected],
        "map_summary": {
            "mean_improvement": float(np.mean([r["improvement"] for r in records])),
            "fraction_improved": float(np.mean([r["improvement"] > 0 for r in records])),
            "high_mismatch_threshold": 0.25,
            "broad_coverage_threshold": 0.55,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("am2022_curves.json"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("ablation_campaign/full_seed20261020/model/model.pt"))
    parser.add_argument("--gate-results", type=Path,
                        default=Path("final_protocol/selective_main/selective_main_metrics.json"))
    parser.add_argument("--database-prefix", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("publication_figures"))
    parser.add_argument("--paper-figures", type=Path, default=Path("paper_elsevier_draft/figs"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("final_protocol/mechanism_case_study_manifest.json"))
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    parser.add_argument("--batch", type=int, default=12)
    main(parser.parse_args())
