#!/usr/bin/env python3
"""Evaluate a trained ControlNet Brownian bridge on every eligible external curve."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from control_bridge import BaseUNet, ControlBridge, generate, make_case
from deep_bridge import load_curves, setup_device, cleanup


def mask_rng(curve_id, known, mask_index, seed):
    token = f"{curve_id}|{known}|{mask_index}|{seed}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.random.default_rng(value)


def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train, val, test = load_curves(args.data, checkpoint["seed"])
    curves = (train + val + test)[rank::world]
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    record_truth, record_prediction, record_known, record_width = [], [], [], []
    # n, |e|, e2, e, y, y2, factor2, factor3, covered, width, violations, grid_edges, seconds, cases
    # HCCL on Ascend does not reduce float64 tensors. These aggregate counts and
    # sums remain well within float32 precision for the present benchmark size.
    stats = torch.zeros((3, 14), dtype=torch.float32, device=device)
    for ki, known in enumerate((2, 3, 4)):
        cases = []
        for curve in curves:
            if len(curve["x"]) < known + 2:
                continue
            for mask_index in range(args.masks):
                rng = mask_rng(curve["id"], known, mask_index, args.seed)
                obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
                cases.append(make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"]))
        for start in range(0, len(cases), args.batch):
            batch = cases[start:start + args.batch]
            names = ("static", "control", "raw", "phys", "mask", "oy")
            static, control, raw, phys, mask, observed = [
                torch.tensor(np.stack([case[name] for case in batch]), device=device) for name in names
            ]
            tic = time.time()
            samples = [
                generate(model, static, control, raw, phys, mask, observed, args.steps, args.eta).cpu().numpy()
                for _ in range(args.ensemble)
            ]
            elapsed = time.time() - tic
            draws = np.stack(samples)
            for index, case in enumerate(batch):
                grid_draws = draws[:, index] * checkpoint["ystd"] + checkpoint["ymean"]
                hidden = np.setdiff1d(np.arange(len(case["y"])), case["obs"])
                point_draws = np.stack([
                    np.interp(case["x"][hidden], case["gx"], grid) for grid in grid_draws
                ])
                truth = case["y"][hidden]
                prediction = point_draws.mean(0)
                error = prediction - truth
                low, high = np.quantile(point_draws, [0.05, 0.95], axis=0)
                mean_grid = grid_draws.mean(0)
                record_truth.extend(truth.tolist())
                record_prediction.extend(prediction.tolist())
                record_known.extend([known] * len(truth))
                record_width.extend((high - low).tolist())
                row = stats[ki]
                row[0] += len(error)
                row[1] += np.abs(error).sum()
                row[2] += np.square(error).sum()
                row[3] += error.sum()
                row[4] += truth.sum()
                row[5] += np.square(truth).sum()
                row[6] += (np.abs(error) <= math.log10(2)).sum()
                row[7] += (np.abs(error) <= math.log10(3)).sum()
                row[8] += ((truth >= low) & (truth <= high)).sum()
                row[9] += (high - low).sum()
                row[10] += (np.diff(mean_grid) > 0).sum()
                row[11] += len(mean_grid) - 1
                row[12] += elapsed / max(len(batch), 1)
                row[13] += 1
    if args.prediction_dir is not None:
        args.prediction_dir.mkdir(parents=True, exist_ok=True)
        safe_name = args.dataset_name.lower().replace(" ", "_").replace("-", "_")
        np.savez_compressed(
            args.prediction_dir / f"{safe_name}_rank{rank}.npz",
            truth=np.asarray(record_truth, np.float32),
            prediction=np.asarray(record_prediction, np.float32),
            known=np.asarray(record_known, np.int16),
            interval_width=np.asarray(record_width, np.float32),
        )
    if world > 1:
        dist.all_reduce(stats)
    if rank == 0:
        output = {"dataset": args.dataset_name, "eligible_curves": len(train) + len(val) + len(test),
                  "checkpoint": str(args.checkpoint), "steps": args.steps, "ensemble": args.ensemble,
                  "known_points": {}}
        for ki, known in enumerate((2, 3, 4)):
            values = stats[ki].cpu().numpy()
            n = max(values[0], 1)
            mean_y = values[4] / n
            total_variance = values[5] - n * mean_y * mean_y
            output["known_points"][str(known)] = {
                "hidden_points": int(values[0]),
                "mae_logN": float(values[1] / n),
                "rmse_logN": float(math.sqrt(values[2] / n)),
                "mean_bias_logN": float(values[3] / n),
                "r2": float(1 - values[2] / max(total_variance, 1e-12)),
                "factor_2_accuracy": float(values[6] / n),
                "factor_3_accuracy": float(values[7] / n),
                "coverage90": float(values[8] / n),
                "width90_logN": float(values[9] / n),
                "monotonic_violation": float(values[10] / max(values[11], 1)),
                "seconds_per_curve_ensemble": float(values[12] / max(values[13], 1)),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--masks", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261004)
    main(parser.parse_args())
