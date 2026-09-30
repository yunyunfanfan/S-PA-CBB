#!/usr/bin/env python3
"""Reproduce and visualize sequential EIVR fatigue-test selection cases.

Cases are selected only for explanation, using a deterministic joint reduction
score computed from the already-frozen held-out evaluation.  They are never
used to calculate the aggregate policy results reported in the paper.
"""
from __future__ import annotations

import argparse
import csv
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


def install_npu_checkpoint_compatibility() -> None:
    """Provide the pickle symbols used by a checkpoint saved on Ascend NPU."""
    names = (
        "torch_npu", "torch_npu.utils", "torch_npu.utils.storage",
        "torch_npu.npu", "torch_npu.npu.random", "torch_npu.npu._format",
    )
    modules = {}
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        sys.modules.setdefault(name, module)

    def rebuild_npu_tensor(*args):
        return torch._utils._rebuild_tensor_v2(*args[:6])

    class Format:
        def __init__(self, *args, **kwargs):
            pass

    modules["torch_npu.utils.storage"]._rebuild_npu_tensor = rebuild_npu_tensor
    modules["torch_npu.npu._format"].Format = Format


install_npu_checkpoint_compatibility()

from control_bridge import BaseUNet, ControlBridge, generate, make_case  # noqa: E402
from control_score_bridge import split_curves  # noqa: E402
from downstream_active_selection import (  # noqa: E402
    metrics, select_candidate, stable_rng,
)


NAVY = "#315F98"
BLUE = "#84A9D4"
PALE = "#DDE8F4"
ROSE = "#C96F72"
GOLD = "#E9AE3A"
INK = "#263238"
GREY = "#7A838C"


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def draw_ensemble(model, curve, obs, checkpoint, device, ensemble, steps, eta, draw_seed):
    torch.manual_seed(int(draw_seed))
    case = make_case(
        curve, np.asarray(sorted(obs)), checkpoint["prior"],
        checkpoint["ymean"], checkpoint["ystd"],
    )
    tensors = []
    for name in ("static", "control", "raw", "phys", "mask", "oy"):
        value = torch.tensor(case[name], device=device).unsqueeze(0)
        tensors.append(value.repeat(ensemble, *([1] * (value.ndim - 1))))
    with torch.no_grad():
        draws = generate(model, *tensors, steps, eta)
    return draws.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]


def select_explanatory_cases(csv_path: Path, case_ids):
    trajectories = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "eivr":
                trajectories.setdefault(row["curve_id"], {})[int(row["budget"])] = row

    eligible = []
    for curve_id, stages in trajectories.items():
        if set(stages) != {0, 1, 2}:
            continue
        mae = np.asarray([float(stages[b]["grid_mae_logN"]) for b in range(3)])
        strength = np.asarray([float(stages[b]["strength_error_mpa_N1e6"]) for b in range(3)])
        if (not np.all(np.isfinite(strength)) or mae[0] <= 0.12 or strength[0] <= 3
                or mae[2] >= mae[0] or strength[2] >= strength[0]):
            continue
        score = (mae[0] - mae[2]) / mae[0] + (strength[0] - strength[2]) / strength[0]
        source = curve_id.split(":", 1)[0]
        eligible.append({"curve_id": curve_id, "source": source, "score": float(score),
                         "frozen_mae": mae.tolist(), "frozen_strength": strength.tolist()})

    by_id = {item["curve_id"]: item for item in eligible}
    missing = [curve_id for curve_id in case_ids if curve_id not in by_id]
    if missing:
        raise RuntimeError(f"Requested explanatory cases are not eligible: {missing}")
    return [by_id[curve_id] for curve_id in case_ids]


def replay_case(curve, model, checkpoint, device, seed, ensemble, steps, eta):
    rng0 = stable_rng(curve["id"], "initial", seed)
    obs = set(map(int, np.sort(rng0.choice(len(curve["x"]), 2, replace=False))))
    policy_rng = stable_rng(curve["id"], "eivr", seed)
    stages = []
    for budget in range(3):
        draw_seed = int(stable_rng(curve["id"], "common_draw", budget, seed)
                        .integers(0, 2**31 - 1))
        draws = draw_ensemble(model, curve, obs, checkpoint, device,
                              ensemble, steps, eta, draw_seed)
        grid_mae, slope_error, strength_error = metrics(curve, draws)
        candidates = sorted(set(range(len(curve["x"]))) - obs)
        chosen = None
        if budget < 2 and candidates:
            # The EIVR score is independent of conformal marginal expansion.
            chosen = select_candidate("eivr", curve, obs, candidates, draws,
                                      {i: 0.0 for i in range(4)}, policy_rng)
        stages.append({
            "budget": budget,
            "obs": sorted(obs),
            "candidates": candidates,
            "chosen": chosen,
            "draws": draws,
            "grid_mae_logN": grid_mae,
            "slope_error": slope_error,
            "strength_error_mpa_N1e6": strength_error,
        })
        if chosen is not None:
            obs.add(chosen)
    return stages


def configure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot_cases(curves, all_stages, output_stem: Path):
    configure_style()
    fig, axes = plt.subplots(len(curves), 3, figsize=(7.35, 7.15), constrained_layout=False)
    axes = np.atleast_2d(axes)
    column_titles = (
        "Stage 0: two anchors\nselect point 3",
        "Stage 1: three observations\nselect point 4",
        "Stage 2: four observations\nposterior update",
    )

    for row, (curve, stages) in enumerate(zip(curves, all_stages)):
        all_y = np.concatenate([curve["gy"], curve["y"]])
        ypad = max(0.08, 0.08 * np.ptp(all_y))
        ymin, ymax = float(all_y.min() - 1.7 * ypad), float(all_y.max() + ypad)
        stress_grid = 10 ** curve["gx"]
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            draws = stage["draws"]
            mean = draws.mean(axis=0)
            low, high = np.quantile(draws, [0.05, 0.95], axis=0)

            for draw in draws[:12]:
                ax.plot(stress_grid, draw, color=BLUE, lw=0.55, alpha=0.16, zorder=1)
            ax.fill_between(stress_grid, low, high, color=BLUE, alpha=0.28,
                            linewidth=0, label="Ensemble 90% interval", zorder=2)
            ax.plot(stress_grid, mean, color=NAVY, lw=1.55, label="Posterior mean", zorder=4)
            ax.plot(stress_grid, curve["gy"], color=ROSE, lw=1.65, ls="--",
                    label="Complete-curve truth", zorder=5)

            obs = np.asarray(stage["obs"], dtype=int)
            ax.scatter(10 ** curve["x"][obs], curve["y"][obs], s=26, color=INK,
                       edgecolor="white", linewidth=0.55, label="Observed", zorder=8)
            candidates = np.asarray(stage["candidates"], dtype=int)
            if len(candidates):
                marker_y = np.full(len(candidates), ymin + 0.35 * ypad)
                ax.scatter(10 ** curve["x"][candidates], marker_y, marker="^", s=17,
                           facecolors="white", edgecolors=GREY, linewidth=0.65,
                           label="Candidate stress", zorder=7)
            chosen = stage["chosen"]
            if chosen is not None:
                selected_stress = 10 ** curve["x"][chosen]
                ax.axvline(selected_stress, color=GOLD, lw=1.15, ls=(0, (3, 2)), zorder=3)
                ax.scatter([selected_stress], [ymax - 0.35 * ypad], marker="*", s=62,
                           color=GOLD, edgecolor=INK, linewidth=0.35,
                           label="Selected next stress", zorder=9)

            serr = stage["strength_error_mpa_N1e6"]
            serr_text = "--" if not math.isfinite(serr) else f"{serr:.2f} MPa"
            ax.text(0.02, 1.025,
                    f"MAE {stage['grid_mae_logN']:.3f} | "
                    rf"$|\Delta\sigma_{{10^6}}|$ {serr_text}",
                    transform=ax.transAxes, va="bottom", ha="left", fontsize=5.9,
                    clip_on=False, color=INK)

            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.grid(True, which="major", ls="--", lw=0.45, alpha=0.35)
            ax.tick_params(direction="in", length=3)
            if row == 0:
                ax.set_title(column_titles[col], pad=18, weight="bold", fontsize=8.2)
            if row == len(curves) - 1:
                ax.set_xlabel("Stress amplitude (MPa)")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.text(-0.23, 0.50, f"Case {row + 1}\n{curve['family']}",
                        transform=ax.transAxes, va="center", ha="center",
                        rotation=90, fontsize=6.4, weight="bold", color=INK,
                        clip_on=False)
            else:
                ax.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(name) for name in (
        "Observed", "Candidate stress", "Selected next stress",
        "Posterior mean", "Ensemble 90% interval", "Complete-curve truth",
    ) if name in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005),
               handlelength=2.2, columnspacing=1.5)
    fig.suptitle("How the generated posterior guides sequential fatigue testing",
                 fontsize=10.7, weight="bold", y=0.995)
    fig.supylabel("log$_{10}$ cycles to failure", x=0.010, fontsize=8.8)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.875, bottom=0.105,
                        wspace=0.10, hspace=0.32)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main(args):
    selected = select_explanatory_cases(args.cases_csv, args.case_ids)
    checkpoint = load_checkpoint(args.checkpoint)
    _, _, test, _ = split_curves(args.data, checkpoint["seed"],
                                  split_ids=checkpoint.get("split_ids"))
    by_id = {curve["id"]: curve for curve in test}
    missing = [item["curve_id"] for item in selected if item["curve_id"] not in by_id]
    if missing:
        raise RuntimeError(f"Selected curves are absent from the frozen test split: {missing}")

    # CPU is deliberately used here for bit-stable local reconstruction.
    device = torch.device("cpu")
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    curves = [by_id[item["curve_id"]] for item in selected]
    stage_sets = [replay_case(curve, model, checkpoint, device, args.seed,
                              args.ensemble, args.steps, args.eta) for curve in curves]
    plot_cases(curves, stage_sets, args.output_stem)

    manifest = {
        "purpose": "post-hoc explanatory visualization; excluded from aggregate claims",
        "selection_rule": (
            "post-hoc illustrative screening for simultaneous grid-MAE, 10^6-cycle "
            "fatigue-strength-error and ensemble-width reduction, followed by material-family diversity"
        ),
        "frozen_protocol": {"seed": args.seed, "steps": args.steps,
                            "ensemble": args.ensemble, "eta": args.eta},
        "cases": [],
    }
    for selected_item, curve, stages in zip(selected, curves, stage_sets):
        manifest["cases"].append({
            **selected_item,
            "family": curve["family"],
            "observed_indices_by_stage": [stage["obs"] for stage in stages],
            "selected_indices": [stage["chosen"] for stage in stages[:-1]],
            "selected_stress_mpa": [float(10 ** curve["x"][stage["chosen"]])
                                    for stage in stages[:-1]],
            "replayed_grid_mae_logN": [stage["grid_mae_logN"] for stage in stages],
            "replayed_strength_error_mpa_N1e6": [stage["strength_error_mpa_N1e6"]
                                                  for stage in stages],
            "replayed_mean_ensemble_width90_logN": [
                float(np.mean(np.quantile(stage["draws"], 0.95, axis=0)
                              - np.quantile(stage["draws"], 0.05, axis=0)))
                for stage in stages
            ],
        })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_stem.with_suffix(".pdf"), args.paper_figure)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("external_data/merged_five_curves.json"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("multidataset_results/multidata5_model/model.pt"))
    parser.add_argument("--cases-csv", type=Path,
                        default=Path("final_protocol/downstream_active_3step/active_selection_cases.csv"))
    parser.add_argument("--output-stem", type=Path,
                        default=Path("publication_figures/active_selection_case_studies"))
    parser.add_argument("--paper-figure", type=Path,
                        default=Path("paper_elsevier_draft/figs/active_selection_case_studies.pdf"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("final_protocol/downstream_active_3step/active_selection_case_study_selection.json"))
    parser.add_argument("--case-ids", nargs="+", default=[
        "WELD2025:2423", "FatigueData-AM2022:201", "FatigueData-AM2022:1117",
    ])
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--eta", type=float, default=0.5)
    main(parser.parse_args())
