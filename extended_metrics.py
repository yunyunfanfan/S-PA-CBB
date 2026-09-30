#!/usr/bin/env python3
"""Evaluate deterministic, interval, and full-curve generative metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from control_bridge import make_case
from control_score_bridge import ScoreBase, ScoreControl, sample, split_curves
from deep_bridge import setup_device, cleanup


LEVELS = (0.50, 0.80, 0.90, 0.95)


def interval_score(y, lo, hi, alpha):
    return (hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)


def empirical_crps(draws, truth):
    """CRPS for draws [ensemble, points] and truth [points]."""
    term1 = np.mean(np.abs(draws - truth[None, :]), axis=0)
    ordered = np.sort(draws, axis=0)
    m = len(ordered)
    weights = 2 * np.arange(1, m + 1) - m - 1
    pair_term = np.sum(weights[:, None] * ordered, axis=0) / (m * m)
    return term1 - pair_term


def curve_energy(draws, truth):
    """Energy score with RMS-normalized Euclidean distance."""
    scale = math.sqrt(draws.shape[1])
    first = np.linalg.norm(draws - truth[None, :], axis=1).mean() / scale
    diffs = draws[:, None, :] - draws[None, :, :]
    second = np.linalg.norm(diffs, axis=2).mean() / scale
    return float(first - 0.5 * second)


def variogram_score(draws, truth, power=0.5):
    i, j = np.triu_indices(draws.shape[1], 1)
    observed = np.abs(truth[i] - truth[j]) ** power
    forecast = np.mean(np.abs(draws[:, i] - draws[:, j]) ** power, axis=0)
    return float(np.mean((observed - forecast) ** 2))


def summarize_point(truth, pred):
    error = pred - truth
    ae = np.abs(error)
    denom = np.sum((truth - truth.mean()) ** 2)
    return {
        "mae_logN": float(ae.mean()),
        "rmse_logN": float(np.sqrt(np.mean(error**2))),
        "median_ae_logN": float(np.median(ae)),
        "p90_ae_logN": float(np.quantile(ae, 0.90)),
        "mean_bias_logN": float(error.mean()),
        "r2": float(1 - np.sum(error**2) / denom),
        "pearson_r": float(np.corrcoef(truth, pred)[0, 1]),
        "factor_2_accuracy": float(np.mean(ae <= math.log10(2))),
        "factor_3_accuracy": float(np.mean(ae <= math.log10(3))),
    }


@torch.no_grad()
def main(args):
    device, rank, _, world = setup_device()
    if world != 1 or rank != 0:
        raise RuntimeError("Run this evaluator on one device.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, curves, _ = split_curves(args.data, ck["seed"], split_ids=ck.get("split_ids"))
    model = ScoreControl(ScoreBase(ck["base"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    rng = np.random.default_rng(args.seed)
    result = {"split": args.name, "steps": args.steps, "eta": args.eta, "ensemble": args.ensemble, "known_points": {}}

    for k in (2, 3, 4):
        bags = {name: [] for name in ("truth", "mean", "base", "crps", "wis")}
        interval_bags = {level: {"inside": [], "width": [], "score": []} for level in LEVELS}
        curve_bags = {name: [] for name in ("energy", "energy_basquin", "variogram", "grid_mae", "slope_error", "monotonic_violation", "diversity")}
        cases = []
        for curve in curves:
            if len(curve["x"]) < k + 2:
                continue
            for _ in range(args.masks):
                q = make_case(curve, np.sort(rng.choice(len(curve["x"]), k, replace=False)), ck["prior"], ck["ymean"], ck["ystd"])
                q["curve"] = curve
                cases.append(q)

        for start in range(0, len(cases), args.batch):
            batch = cases[start : start + args.batch]
            tensors = [torch.tensor(np.stack([q[n] for q in batch]), device=device) for n in ("static", "control", "raw", "phys", "mask", "oy")]
            static, control, raw, phys, mask, oy = tensors
            draws = np.stack([
                sample(model, static, control, raw, phys, mask, oy, args.steps, ck["sigma"], args.eta).cpu().numpy()
                for _ in range(args.ensemble)
            ]) * ck["ystd"] + ck["ymean"]

            for j, q in enumerate(batch):
                posterior = draws[:, j]
                truth_grid = q["curve"]["gy"]
                base_grid = q["phys"] * ck["ystd"] + ck["ymean"]
                hidden = np.setdiff1d(np.arange(len(q["y"])), q["obs"])
                truth = q["y"][hidden]
                values = np.stack([np.interp(q["x"][hidden], q["gx"], draw) for draw in posterior])
                mean = values.mean(axis=0)
                base = np.interp(q["x"][hidden], q["gx"], base_grid)
                bags["truth"].extend(truth)
                bags["mean"].extend(mean)
                bags["base"].extend(base)
                bags["crps"].extend(empirical_crps(values, truth))

                median = np.median(values, axis=0)
                wis_terms = 0.5 * np.abs(truth - median)
                for level in LEVELS:
                    alpha = 1 - level
                    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2], axis=0)
                    score = interval_score(truth, lo, hi, alpha)
                    interval_bags[level]["inside"].extend((truth >= lo) & (truth <= hi))
                    interval_bags[level]["width"].extend(hi - lo)
                    interval_bags[level]["score"].extend(score)
                    wis_terms += (alpha / 2) * score
                bags["wis"].extend(wis_terms / (len(LEVELS) + 0.5))

                curve_bags["energy"].append(curve_energy(posterior, truth_grid))
                curve_bags["energy_basquin"].append(float(np.sqrt(np.mean((base_grid - truth_grid) ** 2))))
                curve_bags["variogram"].append(variogram_score(posterior, truth_grid))
                curve_bags["grid_mae"].append(float(np.mean(np.abs(posterior.mean(axis=0) - truth_grid))))
                slope_true = np.polyfit(q["gx"], truth_grid, 1)[0]
                slope_pred = np.polyfit(q["gx"], posterior.mean(axis=0), 1)[0]
                curve_bags["slope_error"].append(float(abs(slope_pred - slope_true)))
                curve_bags["monotonic_violation"].append(float(np.mean(np.diff(posterior, axis=1) > 0)))
                diffs = posterior[:, None, :] - posterior[None, :, :]
                curve_bags["diversity"].append(float(np.mean(np.abs(diffs))))

        truth = np.asarray(bags["truth"])
        mean = np.asarray(bags["mean"])
        base = np.asarray(bags["base"])
        point_model = summarize_point(truth, mean)
        point_base = summarize_point(truth, base)
        intervals = {}
        for level in LEVELS:
            b = interval_bags[level]
            intervals[str(level)] = {
                "coverage": float(np.mean(b["inside"])),
                "mean_width_logN": float(np.mean(b["width"])),
                "interval_score": float(np.mean(b["score"])),
                "absolute_coverage_error": float(abs(np.mean(b["inside"]) - level)),
            }
        result["known_points"][str(k)] = {
            "points": len(truth),
            "curves": len(cases),
            "point_model": point_model,
            "point_basquin": point_base,
            "mae_skill_over_basquin": float(1 - point_model["mae_logN"] / point_base["mae_logN"]),
            "crps_logN": float(np.mean(bags["crps"])),
            "crps_skill_over_degenerate_basquin": float(1 - np.mean(bags["crps"]) / point_base["mae_logN"]),
            "weighted_interval_score": float(np.mean(bags["wis"])),
            "intervals": intervals,
            "curve_metrics": {name: float(np.mean(values)) for name, values in curve_bags.items()},
            "energy_skill_over_basquin": float(1 - np.mean(curve_bags["energy"]) / np.mean(curve_bags["energy_basquin"])),
        }
        print(json.dumps({"split": args.name, "k": k, "metrics": result["known_points"][str(k)]}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output, flush=True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--masks", type=int, default=1)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261003)
    main(parser.parse_args())
