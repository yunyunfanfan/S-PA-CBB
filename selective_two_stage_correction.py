#!/usr/bin/env python3
"""Validation-calibrated selective residual correction for PA-CBB.

Stage 1 constructs both the sparse Basquin state and the PA-CBB posterior.
Stage 2 uses only quantities available at inference time to decide whether the
posterior-mean residual should be applied.  Complete-curve truth is used only
to select the gate on validation curves and to audit it on held-out test curves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_mechanism_case_studies import (
    BLUE, GOLD, GREY, INK, LIGHT_BLUE, NAVY, ROSE,
    BaseUNet, ControlBridge, evaluate, install_npu_checkpoint_compatibility,
    split_curves, stable_seed, strength_samples, style,
)


def prospective_features(record):
    """Return gate features without accessing the complete reference curve."""
    physics = np.asarray(record["physics"], float)
    draws = np.asarray(record["draws"], float)
    mean = draws.mean(axis=0)
    residual = mean - physics
    generated_linear = np.polyval(np.polyfit(np.arange(len(mean)), mean, 1),
                                  np.arange(len(mean)))
    return {
        "correction": float(np.mean(np.abs(residual))),
        "width": float(record["width90"]),
        "coverage": float(record["coverage"]),
        "signal_to_width": float(np.mean(np.abs(residual)) / (record["width90"] + 1e-4)),
        "curvature": float(np.sqrt(np.mean((mean - generated_linear) ** 2))),
        "roughness": float(np.mean(np.abs(np.diff(residual, n=2)))),
        "endpoint_shift": float(0.5 * (abs(residual[0]) + abs(residual[-1]))),
    }


def candidate_scores(records):
    feats = [prospective_features(r) for r in records]
    score = lambda name: np.asarray([f[name] for f in feats], float)
    correction, width, coverage = score("correction"), score("width"), score("coverage")
    curvature, roughness = score("curvature"), score("roughness")
    endpoint = score("endpoint_shift")
    eps = 1e-4
    return {
        "residual magnitude": correction,
        "residual / spread": correction / (width + eps),
        "support-weighted residual": correction * np.sqrt(coverage + .05) / (width + eps),
        "shape-aware residual": correction * (1 + curvature + roughness) / (width + eps),
        "tail-aware residual": (correction + .5 * endpoint) / (width + eps),
    }


def mixed_mae(records, selected, alpha=1.0):
    values = []
    strength = []
    for r, use in zip(records, selected):
        truth = np.asarray(r["curve"]["gy"], float)
        physics = np.asarray(r["physics"], float)
        generated = np.asarray(r["draws"], float).mean(axis=0)
        pred = physics + (alpha if use else 0.0) * (generated - physics)
        values.append(np.mean(np.abs(pred - truth)))
        estimated_strength = strength_samples(np.asarray(r["curve"]["gx"], float), [pred])[0]
        e = abs(estimated_strength - r["truth_strength"])
        if math.isfinite(e):
            strength.append(e)
    return float(np.mean(values)), float(np.mean(strength)) if strength else float("nan")


def error_arrays(records, alpha):
    grid, strength = [], []
    for r in records:
        truth = np.asarray(r["curve"]["gy"], float)
        physics = np.asarray(r["physics"], float)
        generated = np.asarray(r["draws"], float).mean(axis=0)
        pred = physics + alpha * (generated - physics)
        grid.append(np.mean(np.abs(pred - truth)))
        pred_strength = strength_samples(np.asarray(r["curve"]["gx"], float), [pred])[0]
        strength.append(abs(pred_strength - r["truth_strength"]))
    return np.asarray(grid), np.asarray(strength)


def choose_gate(validation):
    physics = np.asarray([r["physics_mae"] for r in validation])
    generated = np.asarray([r["generated_mae"] for r in validation])
    physics_grid, physics_strength = error_arrays(validation, 0.0)
    finite_strength = np.isfinite(physics_strength)
    base_grid = float(physics_grid.mean())
    base_strength = float(physics_strength[finite_strength].mean())
    alternatives = {alpha: error_arrays(validation, alpha) for alpha in (0.5, 0.75, 1.0)}
    best = {"objective": 2.0, "mae": base_grid, "strength_mae_mpa": base_strength,
            "name": "retain physics", "threshold": float("inf"), "alpha": 0.0, "rate": 0.0}
    scores = candidate_scores(validation)
    for name, values in scores.items():
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 81)))
        for threshold in quantiles:
            mask = values >= threshold
            for alpha in (0.5, 0.75, 1.0):
                alternative_grid, alternative_strength = alternatives[alpha]
                gated_grid = np.where(mask, alternative_grid, physics_grid)
                gated_strength = np.where(mask, alternative_strength, physics_strength)
                valid = np.isfinite(gated_strength) & finite_strength
                mae = float(gated_grid.mean())
                strength_mae = float(gated_strength[valid].mean())
                objective = mae / base_grid + strength_mae / base_strength
                # Require validation improvement in both engineering and curve metrics.
                if mae > base_grid or strength_mae > base_strength:
                    continue
                key = (objective, mae, strength_mae, float(mask.mean()))
                old = (best["objective"], best["mae"], best["strength_mae_mpa"], best["rate"])
                if key < old:
                    best = {"objective": objective, "mae": mae,
                            "strength_mae_mpa": strength_mae, "name": name,
                            "threshold": float(threshold), "alpha": float(alpha),
                            "rate": float(mask.mean())}
    best["validation_physics"] = float(physics.mean())
    best["validation_pacbb"] = float(generated.mean())
    return best


def apply_gate(records, gate):
    values = candidate_scores(records)[gate["name"]]
    return values >= gate["threshold"], values


def bootstrap_delta(records, selected, alpha, repeats=2000, seed=20260929):
    rng = np.random.default_rng(seed)
    physics = np.asarray([r["physics_mae"] for r in records])
    gated = []
    for r, use in zip(records, selected):
        truth = np.asarray(r["curve"]["gy"], float)
        p = np.asarray(r["physics"], float)
        g = np.asarray(r["draws"], float).mean(axis=0)
        pred = p + (alpha if use else 0.0) * (g - p)
        gated.append(np.mean(np.abs(pred - truth)))
    gated = np.asarray(gated)
    # Four layouts from the same curve are dependent, so resample curve IDs.
    curve_ids = np.asarray([r["curve_id"] for r in records])
    unique_ids = np.unique(curve_ids)
    curve_delta = np.asarray([np.mean((gated - physics)[curve_ids == cid]) for cid in unique_ids])
    indices = rng.integers(0, len(unique_ids), size=(repeats, len(unique_ids)))
    delta = np.mean(curve_delta[indices], axis=1)
    return np.quantile(delta, [.025, .5, .975]).tolist()


def summarize(records, selected, gate):
    physics_curve, physics_strength = mixed_mae(records, np.zeros(len(records), bool), 0)
    pacbb_curve, pacbb_strength = mixed_mae(records, np.ones(len(records), bool), 1)
    gated_curve, gated_strength = mixed_mae(records, selected, gate["alpha"])
    physics_err, _ = error_arrays(records, 0.0)
    corrected_err, _ = error_arrays(records, gate["alpha"])
    beneficial = corrected_err < physics_err
    precision = float(beneficial[selected].mean()) if selected.any() else float("nan")
    recall = float(selected[beneficial].mean()) if beneficial.any() else float("nan")
    oracle = float(np.minimum(physics_err, corrected_err).mean())
    return {
        "Basquin": {"grid_mae": physics_curve, "strength_mae_mpa": physics_strength},
        "always_PA_CBB": {"grid_mae": pacbb_curve, "strength_mae_mpa": pacbb_strength},
        "selective_PA_CBB": {"grid_mae": gated_curve, "strength_mae_mpa": gated_strength},
        "oracle_selective_weight": {"grid_mae": oracle},
        "activation_rate": float(selected.mean()),
        "selection_precision": precision,
        "selection_recall": recall,
        "bootstrap_delta_vs_Basquin": bootstrap_delta(records, selected, gate["alpha"]),
    }


def by_layout(records, selected, gate):
    rows = []
    for layout in sorted({r["layout"] for r in records}):
        idx = np.asarray([r["layout"] == layout for r in records])
        subset = [r for r, keep in zip(records, idx) if keep]
        choice = selected[idx]
        summary = summarize(subset, choice, gate)
        rows.append({"layout": layout,
                     "Basquin": summary["Basquin"]["grid_mae"],
                     "always_PA_CBB": summary["always_PA_CBB"]["grid_mae"],
                     "selective_PA_CBB": summary["selective_PA_CBB"]["grid_mae"],
                     "activation_rate": summary["activation_rate"]})
    return rows


def plot_results(validation, test, gate, selected, scores, summary, layout_rows, stem):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.7))
    physics_error, _ = error_arrays(test, 0.0)
    corrected_error, _ = error_arrays(test, gate["alpha"])
    gain = physics_error - corrected_error
    ax = axes[0, 0]
    ax.scatter(scores, gain, c=np.where(selected, NAVY, GREY), s=15, alpha=.55,
               edgecolor="none")
    ax.axvline(gate["threshold"], color=GOLD, lw=1.6, ls="--", label="validation threshold")
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xlabel(gate["name"]); ax.set_ylabel("Basquin - PA-CBB MAE")
    ax.set_title("(a) Held-out gate behaviour", loc="left", weight="bold")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    names = ["Basquin", "Always\nPA-CBB", "Selective\nPA-CBB", "Oracle\nswitch"]
    values = [summary["Basquin"]["grid_mae"], summary["always_PA_CBB"]["grid_mae"],
              summary["selective_PA_CBB"]["grid_mae"], summary["oracle_selective_weight"]["grid_mae"]]
    bars = ax.bar(names, values, color=[GREY, ROSE, NAVY, LIGHT_BLUE], edgecolor=INK, linewidth=.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Grid MAE (log cycles)"); ax.set_title("(b) Independent test result", loc="left", weight="bold")

    ax = axes[1, 0]
    layouts = [r["layout"].replace("_", "\n") for r in layout_rows]
    x = np.arange(len(layouts)); width = .25
    for j, (key, color, label) in enumerate((("Basquin", GREY, "Basquin"),
                                             ("always_PA_CBB", ROSE, "Always PA-CBB"),
                                             ("selective_PA_CBB", NAVY, "Selective PA-CBB"))):
        ax.bar(x + (j-1)*width, [r[key] for r in layout_rows], width, color=color, label=label)
    ax.set_xticks(x, layouts); ax.set_ylabel("Grid MAE (log cycles)")
    ax.set_title("(c) Robustness to sparse-point layout", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=6.2)

    ax = axes[1, 1]
    ax.bar(["Activated", "Retained\nphysics"], [selected.mean(), 1-selected.mean()],
           color=[NAVY, LIGHT_BLUE], edgecolor=INK, linewidth=.5)
    ax.text(.5, .90, f"selection precision = {summary['selection_precision']:.1%}\n"
            f"selection recall = {summary['selection_recall']:.1%}\n"
            f"correction weight = {gate['alpha']:.2f}", transform=ax.transAxes,
            ha="center", va="top", bbox=dict(fc="white", ec=LIGHT_BLUE, boxstyle="round,pad=.35"))
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction of held-out cases")
    ax.set_title("(d) Selective-use audit", loc="left", weight="bold")

    for ax in axes.ravel():
        ax.grid(axis="y", ls="--", lw=.45, alpha=.3); ax.tick_params(direction="in")
    fig.suptitle("Two-stage selective physical-to-generative correction", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=1.4, w_pad=1.2)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main(args):
    install_npu_checkpoint_compatibility()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, validation_curves, test_curves, _ = split_curves(
        args.data, checkpoint["split_seed"], split_ids=checkpoint["split_ids"])
    device = torch.device("cpu")
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    layouts = ("random", "endpoints", "high_stress", "low_stress")
    validation = evaluate(validation_curves, model, checkpoint, device, layouts,
                          args.ensemble, args.steps, args.eta, args.mask_seed, args.batch)
    test = evaluate(test_curves, model, checkpoint, device, layouts,
                    args.ensemble, args.steps, args.eta, args.mask_seed, args.batch)
    gate = choose_gate(validation)
    selected, scores = apply_gate(test, gate)
    summary = summarize(test, selected, gate)
    layout_rows = by_layout(test, selected, gate)

    output = {"protocol": {"validation_curves": len(validation_curves),
                            "test_curves": len(test_curves), "layouts": list(layouts),
                            "validation_cases": len(validation), "test_cases": len(test),
                            "steps": args.steps, "ensemble": args.ensemble, "eta": args.eta},
              "gate": gate, "test": summary, "by_layout": layout_rows,
              "feature_policy": "all gate variables are available before complete-curve truth"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selective_two_stage_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    with (args.output_dir / "selective_two_stage_by_layout.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=layout_rows[0].keys())
        writer.writeheader(); writer.writerows(layout_rows)
    stem = args.output_dir / "selective_two_stage_correction"
    plot_results(validation, test, gate, selected, scores, summary, layout_rows, stem)
    args.paper_figures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stem.with_suffix(".pdf"), args.paper_figures / stem.with_suffix(".pdf").name)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("am2022_curves.json"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("ablation_campaign/full_seed20261020/model/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("final_protocol/selective_two_stage"))
    parser.add_argument("--paper-figures", type=Path, default=Path("paper_elsevier_draft/figs"))
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    parser.add_argument("--batch", type=int, default=12)
    main(parser.parse_args())
