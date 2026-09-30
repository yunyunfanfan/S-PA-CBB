#!/usr/bin/env python3
"""Nested posterior-size sweep for fatigue-strength retrieval.

Each sparse condition is generated once with 64 draws; smaller ensembles are
strict prefixes, so changes cannot be attributed to different bridge noise.
"""
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
from downstream_fatigue_strength import (TARGETS, invert_log_stress,
                                         monotone_decreasing, pairwise_accuracy,
                                         spearman)


SIZES = (1, 4, 8, 16, 32, 64)


def stable_seed(*parts) -> int:
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31 - 1)


def conformal_quantile(values, alpha=0.1):
    if not values:
        return 0.0
    level = min(1.0, math.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


@torch.no_grad()
def evaluate_split(curves, split_name, model, checkpoint, device, args, rank, world):
    records = []
    for curve in curves[rank::world]:
        for known in (2, 3, 4):
            if len(curve["x"]) < known + 2:
                continue
            rng = np.random.default_rng(stable_seed(curve["id"], known, args.mask_seed))
            obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
            case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
            tensors = []
            for name in ("static", "control", "raw", "phys", "mask", "oy"):
                value = torch.tensor(case[name], device=device).unsqueeze(0)
                tensors.append(value.repeat(max(SIZES), *([1] * (value.ndim - 1))))
            draw_seed = stable_seed(curve["id"], known, split_name, args.draw_seed)
            torch.manual_seed(draw_seed)
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.manual_seed_all(draw_seed)
            draws = generate(model, *tensors, args.steps, args.eta).cpu().numpy()
            draws = draws * checkpoint["ystd"] + checkpoint["ymean"]
            reference = monotone_decreasing(curve["gy"])
            for target in TARGETS:
                if not (reference.min() <= target <= reference.max()):
                    continue
                true_x = invert_log_stress(curve["gx"], reference, target)
                all_x = np.asarray([invert_log_stress(curve["gx"], draw, target) for draw in draws])
                for size in SIZES:
                    sample = all_x[:size]
                    low, high = np.quantile(sample, [0.05, 0.95])
                    records.append({
                        "split": split_name, "curve_id": curve["id"], "known": known,
                        "target_logN": target, "ensemble": size, "true_logS": true_x,
                        "pred_logS": float(sample.mean()), "raw_lo_logS": float(low),
                        "raw_hi_logS": float(high),
                    })
    return records


def summarize(validation, test):
    calibration = {}
    for size in SIZES:
        for known in (2, 3, 4):
            for target in TARGETS:
                rows = [r for r in validation if r["ensemble"] == size and
                        r["known"] == known and r["target_logN"] == target]
                scores = [max(r["raw_lo_logS"] - r["true_logS"],
                              r["true_logS"] - r["raw_hi_logS"], 0.0) for r in rows]
                calibration[(size, known, target)] = conformal_quantile(scores)

    detail = []
    for row in test:
        item = dict(row); q = calibration[(row["ensemble"], row["known"], row["target_logN"])]
        item["calibration_q_logS"] = q
        item["lo_logS"] = row["raw_lo_logS"] - q
        item["hi_logS"] = row["raw_hi_logS"] + q
        for key in ("true", "pred", "raw_lo", "raw_hi", "lo", "hi"):
            item[f"{key}_stress_mpa"] = 10 ** item[f"{key}_logS"]
        detail.append(item)

    summary = []
    for size in SIZES:
        for known in (2, 3, 4):
            for target in TARGETS:
                rows = [r for r in detail if r["ensemble"] == size and
                        r["known"] == known and r["target_logN"] == target]
                if not rows:
                    continue
                true = np.asarray([r["true_stress_mpa"] for r in rows])
                pred = np.asarray([r["pred_stress_mpa"] for r in rows])
                raw_lo = np.asarray([r["raw_lo_stress_mpa"] for r in rows])
                raw_hi = np.asarray([r["raw_hi_stress_mpa"] for r in rows])
                lo = np.asarray([r["lo_stress_mpa"] for r in rows])
                hi = np.asarray([r["hi_stress_mpa"] for r in rows])
                summary.append({
                    "ensemble": size, "known": known, "target_cycles": int(10 ** target),
                    "cases": len(rows), "stress_mae_mpa": float(np.mean(np.abs(pred - true))),
                    "relative_stress_error": float(np.mean(np.abs(pred - true) / np.maximum(true, 1e-9))),
                    "spearman_rho": spearman(true, pred),
                    "raw_coverage90": float(np.mean((true >= raw_lo) & (true <= raw_hi))),
                    "conformal_coverage90": float(np.mean((true >= lo) & (true <= hi))),
                    "raw_interval_width_mpa": float(np.mean(raw_hi - raw_lo)),
                    "conformal_interval_width_mpa": float(np.mean(hi - lo)),
                    "pairwise_ranking_accuracy": pairwise_accuracy(true, pred),
                })
    return detail, summary


def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split_seed = checkpoint.get("split_seed", checkpoint.get("seed", args.split_seed))
    _, validation, test, _ = split_curves(args.data, split_seed,
                                           split_ids=checkpoint.get("split_ids"))
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val = evaluate_split(validation, "validation", model, checkpoint, device, args, rank, world)
    tst = evaluate_split(test, "test", model, checkpoint, device, args, rank, world)
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps({"validation": val, "test": tst}))
    if world > 1:
        dist.barrier()
    if rank == 0:
        validation_all, test_all = [], []
        for index in range(world):
            payload = json.loads((args.output_dir / f"rank{index}.json").read_text())
            validation_all.extend(payload["validation"]); test_all.extend(payload["test"])
        detail, summary = summarize(validation_all, test_all)
        output = {"steps": args.steps, "eta": args.eta, "ensemble_sizes": list(SIZES),
                  "nested_draws": True, "mask_seed": args.mask_seed, "summary": summary}
        (args.output_dir / "strength_ensemble_sweep.json").write_text(json.dumps(output, indent=2))
        with (args.output_dir / "strength_ensemble_sweep.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        with (args.output_dir / "strength_ensemble_cases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
        print(json.dumps(output, indent=2))
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    parser.add_argument("--draw-seed", type=int, default=20261101)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
