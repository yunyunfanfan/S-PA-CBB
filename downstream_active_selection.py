#!/usr/bin/env python3
"""Offline active fatigue-test selection on untouched held-out curves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from control_bridge import BaseUNet, ControlBridge, generate, make_case
from control_score_bridge import split_curves
from deep_bridge import cleanup, setup_device
from downstream_fatigue_strength import invert_log_stress, monotone_decreasing


POLICIES = ("random", "largest_gap", "space_filling", "raw_variance", "conformal_width", "eivr")


def stable_rng(*parts) -> np.random.Generator:
    token = "|".join(map(str, parts)).encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(seed)


@torch.no_grad()
def draw_ensemble(model, curve, obs, checkpoint, device, args, draw_seed=None):
    # Common random numbers make policy comparisons paired: when two policies
    # have the same observations at a budget, they receive the same bridge noise.
    if draw_seed is not None:
        torch.manual_seed(int(draw_seed))
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed_all(int(draw_seed))
    case = make_case(curve, np.asarray(sorted(obs)), checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
    tensors = []
    for name in ("static", "control", "raw", "phys", "mask", "oy"):
        value = torch.tensor(case[name], device=device).unsqueeze(0)
        tensors.append(value.repeat(args.ensemble, *([1] * (value.ndim - 1))))
    draws = generate(model, *tensors, args.steps, args.eta)
    return draws.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]


def position_bin(curve, x, bins=4):
    value = (x - curve["gx"].min()) / max(np.ptp(curve["gx"]), 1e-8)
    return min(bins - 1, max(0, int(value * bins)))


def calibration_by_position(validation, model, checkpoint, device, args, rank, world):
    local_scores = {index: [] for index in range(4)}
    for curve in validation[rank::world]:
        if len(curve["x"]) < 5:
            continue
        rng = stable_rng(curve["id"], "calibration", args.seed)
        obs = np.sort(rng.choice(len(curve["x"]), 2, replace=False))
        draw_seed = int(stable_rng(curve["id"], "calibration_draw", args.seed).integers(0, 2**31 - 1))
        draws = draw_ensemble(model, curve, obs, checkpoint, device, args, draw_seed)
        hidden = np.setdiff1d(np.arange(len(curve["x"])), obs)
        for index in hidden:
            values = np.asarray([np.interp(curve["x"][index], curve["gx"], draw) for draw in draws])
            low, high = np.quantile(values, [0.05, 0.95])
            truth = curve["y"][index]
            score = max(low - truth, truth - high, 0.0)
            local_scores[position_bin(curve, curve["x"][index])].append(float(score))
    part = args.output_dir / f"active_calibration_rank{rank}.json"
    part.write_text(json.dumps(local_scores))
    if world > 1:
        dist.barrier()
    if rank != 0:
        return None
    merged = {index: [] for index in range(4)}
    for r in range(world):
        payload = json.loads((args.output_dir / f"active_calibration_rank{r}.json").read_text())
        for index in range(4):
            merged[index].extend(payload[str(index)])
    quantiles = {}
    for index, values in merged.items():
        level = min(1.0, math.ceil((len(values) + 1) * 0.9) / max(len(values), 1))
        quantiles[index] = float(np.quantile(values, level, method="higher")) if values else 0.0
    (args.output_dir / "active_calibration.json").write_text(json.dumps(quantiles, indent=2))
    return quantiles


def select_candidate(policy, curve, obs, candidates, draws, conformal_q, rng):
    xobs = curve["x"][list(obs)]
    if policy == "random":
        return int(rng.choice(candidates))
    if policy == "largest_gap":
        ordered = np.sort(xobs)
        bounds = np.concatenate([[curve["gx"].min()], ordered, [curve["gx"].max()]])
        gap_index = int(np.argmax(np.diff(bounds)))
        midpoint = (bounds[gap_index] + bounds[gap_index + 1]) / 2
        inside = [i for i in candidates if bounds[gap_index] <= curve["x"][i] <= bounds[gap_index + 1]]
        pool = inside or list(candidates)
        return int(min(pool, key=lambda i: abs(curve["x"][i] - midpoint)))
    if policy == "space_filling":
        return int(max(candidates, key=lambda i: np.min(np.abs(curve["x"][i] - xobs))))

    candidate_draws = np.asarray([[np.interp(curve["x"][i], curve["gx"], draw)
                                   for draw in draws] for i in candidates])
    if policy == "raw_variance":
        return int(candidates[int(np.argmax(candidate_draws.var(axis=1)))])
    if policy == "conformal_width":
        low, high = np.quantile(candidate_draws, [0.05, 0.95], axis=1)
        width = high - low + np.asarray([2 * conformal_q[position_bin(curve, curve["x"][i])]
                                         for i in candidates])
        return int(candidates[int(np.argmax(width))])
    if policy == "eivr":
        # Gaussian conditional-variance approximation to expected integrated
        # variance reduction.  It uses the complete joint curve ensemble.
        centered_grid = draws - draws.mean(axis=0, keepdims=True)
        scores = []
        for values in candidate_draws:
            centered = values - values.mean()
            variance = np.mean(centered ** 2) + 1e-6
            covariance = np.mean(centered_grid * centered[:, None], axis=0)
            scores.append(float(np.mean(covariance ** 2) / variance))
        return int(candidates[int(np.argmax(scores))])
    raise ValueError(policy)


def metrics(curve, draws):
    mean = draws.mean(axis=0)
    grid_mae = float(np.mean(np.abs(mean - curve["gy"])))
    slope = float(abs(np.polyfit(curve["gx"], mean, 1)[0] -
                      np.polyfit(curve["gx"], curve["gy"], 1)[0]))
    strength_error = float("nan")
    reference = monotone_decreasing(curve["gy"])
    if reference.min() <= 6.0 <= reference.max():
        truth = 10 ** invert_log_stress(curve["gx"], reference, 6.0)
        predictions = [10 ** invert_log_stress(curve["gx"], draw, 6.0) for draw in draws]
        strength_error = float(abs(np.mean(predictions) - truth))
    return grid_mae, slope, strength_error


def evaluate(curves, model, checkpoint, device, args, rank, world, conformal_q):
    records = []
    for curve in curves[rank::world]:
        if len(curve["x"]) < 6:
            continue
        initial_rng = stable_rng(curve["id"], "initial", args.seed)
        initial = set(map(int, np.sort(initial_rng.choice(len(curve["x"]), 2, replace=False))))
        for policy in POLICIES:
            obs = set(initial); trajectory = []
            rng = stable_rng(curve["id"], policy, args.seed)
            for budget in range(3):
                common_seed = int(stable_rng(curve["id"], "common_draw", budget, args.seed)
                                  .integers(0, 2**31 - 1))
                draws = draw_ensemble(model, curve, obs, checkpoint, device, args, common_seed)
                grid_mae, slope_error, strength_error = metrics(curve, draws)
                trajectory.append((grid_mae, slope_error, strength_error))
                records.append({
                    "curve_id": curve["id"], "policy": policy, "budget": budget,
                    "known_points": len(obs), "grid_mae_logN": grid_mae,
                    "slope_error": slope_error, "strength_error_mpa_N1e6": strength_error,
                })
                if budget == 2:
                    break
                candidates = sorted(set(range(len(curve["x"]))) - obs)
                if not candidates:
                    break
                chosen = select_candidate(policy, curve, obs, candidates, draws, conformal_q, rng)
                obs.add(chosen)  # reveal the actual held-out experimental value
    return records


def aggregate(records, threshold):
    by_curve_policy = {}
    for row in records:
        by_curve_policy.setdefault((row["curve_id"], row["policy"]), {})[row["budget"]] = row
    summary = []
    for policy in POLICIES:
        groups = [trajectory for (curve_id, name), trajectory in by_curve_policy.items()
                  if name == policy and all(b in trajectory for b in (0, 1, 2))]
        if not groups:
            continue
        mae = np.asarray([[g[b]["grid_mae_logN"] for b in (0, 1, 2)] for g in groups])
        slopes = np.asarray([[g[b]["slope_error"] for b in (0, 1, 2)] for g in groups])
        strength = np.asarray([[g[b]["strength_error_mpa_N1e6"] for b in (0, 1, 2)] for g in groups])
        auc = np.trapz(mae, x=np.asarray([0, 1, 2]), axis=1) / 2.0
        tests = []
        for values in mae:
            hit = np.flatnonzero(values <= threshold)
            tests.append(int(hit[0]) if len(hit) else 3)
        summary.append({
            "policy": policy, "curves": len(groups),
            "grid_mae_budget_auc": float(auc.mean()),
            "mae_after_one_test": float(mae[:, 1].mean()),
            "mae_after_two_tests": float(mae[:, 2].mean()),
            "slope_error_after_two_tests": float(slopes[:, 2].mean()),
            "strength_error_mpa_N1e6_after_two_tests": float(np.nanmean(strength[:, 2])),
            "mean_tests_to_threshold": float(np.mean(tests)),
            "threshold_success_rate": float(np.mean(np.asarray(tests) <= 2)),
        })
    return summary


def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split_seed = checkpoint.get("split_seed", checkpoint.get("seed", args.split_seed))
    train, validation, test, _ = split_curves(args.data, split_seed,
                                               split_ids=checkpoint.get("split_ids"))
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quantiles = calibration_by_position(validation, model, checkpoint, device, args, rank, world)
    if world > 1:
        dist.barrier()
    if rank != 0:
        quantiles = json.loads((args.output_dir / "active_calibration.json").read_text())
        quantiles = {int(k): float(v) for k, v in quantiles.items()}
    records = evaluate(test, model, checkpoint, device, args, rank, world, quantiles)
    (args.output_dir / f"active_selection_rank{rank}.json").write_text(json.dumps(records))
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for r in range(world):
            merged.extend(json.loads((args.output_dir / f"active_selection_rank{r}.json").read_text()))
        summary = aggregate(merged, args.threshold)
        output = {
            "method": "PA-CBB", "checkpoint": str(args.checkpoint), "split_seed": split_seed,
            "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "steps": args.steps, "ensemble": args.ensemble, "eta": args.eta,
            "grid_mae_threshold": args.threshold,
            "eivr": "Gaussian conditional-variance approximation from the joint generated curve ensemble",
            "summary": summary,
        }
        (args.output_dir / "active_selection_summary.json").write_text(json.dumps(output, indent=2))
        with (args.output_dir / "active_selection_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        with (args.output_dir / "active_selection_cases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(merged[0])); writer.writeheader(); writer.writerows(merged)
        print(json.dumps(output, indent=2), flush=True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
