#!/usr/bin/env python3
"""Remaining source-domain sensitivity studies under the frozen sampler.

The script evaluates (i) Basquin-slope shrinkage, (ii) sparse-anchor location,
and (iii) a hard monotone post-projection. It deliberately keeps checkpoint,
test curves, reverse steps, posterior sample count and transition noise fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ablation_control_variants import sample
from control_bridge import BaseUNet, ControlBridge, make_case
from control_score_bridge import split_curves
from deep_bridge import cleanup, fit_line, setup_device
from extended_metrics import curve_energy, empirical_crps


RIDGES = (("no_shrink", 0.0), ("weak_shrink", 0.001),
          ("current_shrink", 0.01), ("strong_shrink", 0.1))
MASKS = ("random", "uniform_interior", "high_stress", "low_stress",
         "adjacent_center", "endpoints")


def stable_seed(*parts) -> int:
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31 - 1)


def choose_observations(curve, known: int, scenario: str, seed: int) -> np.ndarray:
    n = len(curve["x"]); order = np.argsort(curve["x"])
    if scenario == "random":
        rng = np.random.default_rng(stable_seed(curve["id"], known, seed))
        return np.sort(rng.choice(n, known, replace=False))
    if scenario == "high_stress":
        return np.sort(order[-known:])
    if scenario == "low_stress":
        return np.sort(order[:known])
    if scenario == "adjacent_center":
        start = max(0, (n - known) // 2)
        return np.sort(order[start:start + known])
    if scenario == "uniform_interior":
        positions = np.rint(np.linspace(0, n - 1, known + 2)[1:-1]).astype(int)
        return np.sort(order[np.unique(positions)])
    if scenario == "endpoints":
        positions = np.rint(np.linspace(0, n - 1, known)).astype(int)
        return np.sort(order[np.unique(positions)])
    raise ValueError(scenario)


def case_with_ridge(curve, obs, checkpoint, ridge):
    case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
    a, b = fit_line(curve["x"][obs], curve["y"][obs], checkpoint["prior"], ridge=ridge)
    physical = a + b * curve["gx"]
    physical_n = (physical - checkpoint["ymean"]) / checkpoint["ystd"]
    case["phys"] = physical_n.astype(np.float32)
    case["control"][0] = case["phys"]
    # The penultimate six raw values are [prior, fitted slope, mean x, span, k, R].
    case["raw"][-5] = np.float32(b / 10)
    return case


def hard_monotone(draws, mask, observed, ymean, ystd):
    """Weighted L2 isotonic projection; anchors receive large but finite weight."""
    projected = []
    weights = np.ones(draws.shape[1], dtype=np.float64)
    anchor = mask.astype(bool); weights[anchor] = 1e4
    anchor_values = observed * ystd + ymean

    def decreasing_pava(values, sample_weight):
        # Pool-adjacent-violators on -values gives a non-increasing L2 fit.
        source = -np.asarray(values, dtype=np.float64)
        levels, block_weights, starts, ends = [], [], [], []
        for index, (value, weight) in enumerate(zip(source, sample_weight)):
            levels.append(float(value)); block_weights.append(float(weight))
            starts.append(index); ends.append(index + 1)
            while len(levels) >= 2 and levels[-2] > levels[-1]:
                total = block_weights[-2] + block_weights[-1]
                pooled = (levels[-2] * block_weights[-2] + levels[-1] * block_weights[-1]) / total
                levels[-2:] = [pooled]; block_weights[-2:] = [total]
                ends[-2:] = [ends[-1]]; starts.pop()
        fitted = np.empty_like(source)
        for level, start, end in zip(levels, starts, ends):
            fitted[start:end] = level
        return -fitted

    for draw in draws:
        values = draw.copy()
        values[anchor] = anchor_values[anchor]
        projected.append(decreasing_pava(values, weights))
    return np.asarray(projected)


def record_metrics(curve, obs, draws, axis, variant, known, anchor_shift=0.0):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
    values = np.asarray([[np.interp(curve["x"][i], curve["gx"], draw)
                          for i in hidden] for draw in draws])
    truth = curve["y"][hidden]; mean = values.mean(axis=0)
    lo, hi = np.quantile(values, [0.05, 0.95], axis=0)
    mean_grid = draws.mean(axis=0)
    xmin, xmax = curve["x"][obs].min(), curve["x"][obs].max()
    interpolation = (curve["x"][hidden] >= xmin) & (curve["x"][hidden] <= xmax)
    extrapolation = ~interpolation

    def subset_mae(select):
        return float(np.mean(np.abs(mean[select] - truth[select]))) if np.any(select) else None

    return {
        "curve_id": curve["id"], "axis": axis, "variant": variant, "known": known,
        "observed_indices": [int(i) for i in obs],
        "mae_logN": float(np.mean(np.abs(mean - truth))),
        "interpolation_mae_logN": subset_mae(interpolation),
        "extrapolation_mae_logN": subset_mae(extrapolation),
        "interpolation_points": int(interpolation.sum()),
        "extrapolation_points": int(extrapolation.sum()),
        "grid_mae_logN": float(np.mean(np.abs(mean_grid - curve["gy"]))),
        "slope_error": float(abs(np.polyfit(curve["gx"], mean_grid, 1)[0] -
                                  np.polyfit(curve["gx"], curve["gy"], 1)[0])),
        "monotonic_violation": float(np.mean(np.diff(draws, axis=1) > 0)),
        "crps_logN": float(np.mean(empirical_crps(values, truth))),
        "energy_score": float(curve_energy(draws, curve["gy"])),
        "coverage90": float(np.mean((truth >= lo) & (truth <= hi))),
        "anchor_shift_logN": float(anchor_shift),
    }


def summarize(records):
    metrics = ("mae_logN", "interpolation_mae_logN", "extrapolation_mae_logN",
               "grid_mae_logN", "slope_error", "monotonic_violation",
               "crps_logN", "energy_score", "coverage90", "anchor_shift_logN")
    output = []
    keys = sorted({(r["axis"], r["variant"], r["known"]) for r in records})
    for axis, variant, known in keys:
        rows = [r for r in records if (r["axis"], r["variant"], r["known"]) == (axis, variant, known)]
        item = {"axis": axis, "variant": variant, "known": known, "curves": len(rows)}
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric] is not None and np.isfinite(r[metric])]
            item[metric] = float(np.mean(values)) if values else None
        item["interpolation_points"] = int(sum(r["interpolation_points"] for r in rows))
        item["extrapolation_points"] = int(sum(r["extrapolation_points"] for r in rows))
        output.append(item)
    return output


@torch.no_grad()
def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, test, _ = split_curves(args.data, checkpoint["split_seed"],
                                  split_ids=checkpoint["split_ids"])
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    local = test[rank::world]; records = []

    configurations = [("shrinkage", name, "random", ridge) for name, ridge in RIDGES]
    configurations += [("mask_location", scenario, scenario, 0.01) for scenario in MASKS]
    # De-duplicate the current/random pair while keeping both semantic axes.
    for known in (2, 3, 4):
        for axis, variant, scenario, ridge in configurations:
            cases = []
            for curve in local:
                if len(curve["x"]) < known + 2:
                    continue
                obs = choose_observations(curve, known, scenario, args.mask_seed)
                if len(obs) != known:
                    continue
                cases.append((curve, obs, case_with_ridge(curve, obs, checkpoint, ridge)))
            for start in range(0, len(cases), args.batch):
                batch = cases[start:start + args.batch]
                tensors = []
                for name in ("static", "control", "raw", "phys", "mask", "oy"):
                    value = torch.tensor(np.stack([case[name] for _, _, case in batch]), device=device)
                    tensors.append(value.repeat_interleave(args.ensemble, dim=0))
                torch.manual_seed(stable_seed(checkpoint["seed"], axis, variant, known, args.draw_seed))
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.manual_seed_all(stable_seed(checkpoint["seed"], axis, variant, known, args.draw_seed))
                normalized = sample(model, "full", *tensors, args.steps, args.eta)
                draws = (normalized.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"])
                draws = draws.reshape(len(batch), args.ensemble, -1)
                for (curve, obs, case), curve_draws in zip(batch, draws):
                    records.append(record_metrics(curve, obs, curve_draws, axis, variant, known))
                    if axis == "shrinkage" and variant == "current_shrink":
                        projected = hard_monotone(curve_draws, case["mask"], case["oy"],
                                                   checkpoint["ymean"], checkpoint["ystd"])
                        anchor = case["mask"].astype(bool)
                        original_anchor = case["oy"][anchor] * checkpoint["ystd"] + checkpoint["ymean"]
                        shift = float(np.mean(np.abs(projected[:, anchor] - original_anchor[None, :])))
                        records.append(record_metrics(curve, obs, projected, "monotonicity",
                                                      "hard_projection", known, shift))
                        records.append(record_metrics(curve, obs, curve_draws, "monotonicity",
                                                      "soft_shape_loss", known, 0.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps(records))
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for index in range(world):
            merged.extend(json.loads((args.output_dir / f"rank{index}.json").read_text()))
        payload = {
            "checkpoint": str(args.checkpoint), "seed": checkpoint["seed"],
            "steps": args.steps, "ensemble": args.ensemble, "eta": args.eta,
            "mask_seed": args.mask_seed, "shrinkage_ridges": dict(RIDGES),
            "summary": summarize(merged), "case_records": merged,
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({key: value for key, value in payload.items() if key != "case_records"}, indent=2))
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    parser.add_argument("--draw-seed", type=int, default=20261031)
    parser.add_argument("--batch", type=int, default=16)
    main(parser.parse_args())
