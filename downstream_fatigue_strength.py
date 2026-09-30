#!/usr/bin/env python3
"""Fatigue-strength inversion from conditional PA-CBB curve ensembles.

The script uses the checkpoint's own split seed/IDs, calibrates 90% intervals on
the validation curves in log-stress space, and evaluates the untouched test
curves.  Every rank writes auditable case-level records before rank zero builds
the aggregate JSON/CSV tables.
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


TARGETS = (5.0, 6.0, 7.0)


def stable_rng(curve_id: str, k: int, mask: int, seed: int) -> np.random.Generator:
    token = f"{curve_id}|{k}|{mask}|{seed}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """Simple monotone rearrangement used only for numerical inversion."""
    return np.minimum.accumulate(np.asarray(values, dtype=np.float64))


def invert_log_stress(gx: np.ndarray, gy: np.ndarray, target: float) -> float:
    curve = monotone_decreasing(gy)
    # np.interp needs an increasing abscissa.  Repeated plateaus are collapsed.
    life = curve[::-1]
    stress = np.asarray(gx, dtype=np.float64)[::-1]
    unique_life, unique_index = np.unique(life, return_index=True)
    unique_stress = stress[unique_index]
    if len(unique_life) == 1:
        return float(unique_stress[0])
    return float(np.interp(target, unique_life, unique_stress,
                           left=unique_stress[0], right=unique_stress[-1]))


def dataset_name(curve_id: str) -> str:
    key = curve_id.lower()
    if "am2022" in key or "fatiguedata-am" in key:
        return "AM2022"
    if "cma2022" in key or "fatiguedata-cma" in key:
        return "CMA2022"
    if "hea2022" in key or key.startswith("hea"):
        return "HEA2022"
    if "weld" in key:
        return "Weld2025"
    if "nims" in key:
        return "NIMS-derived"
    return curve_id.split(":", 1)[0]


@torch.no_grad()
def ensemble(model, case, checkpoint, device, steps, eta, size):
    names = ("static", "control", "raw", "phys", "mask", "oy")
    tensors = []
    for name in names:
        value = torch.tensor(case[name], device=device).unsqueeze(0)
        tensors.append(value.repeat(size, *([1] * (value.ndim - 1))))
    draws = generate(model, *tensors, steps, eta)
    return draws.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]


def evaluate_split(curves, model, checkpoint, device, args, rank, split_name):
    records = []
    local = curves[rank::args.world]
    for curve in local:
        for known in (2, 3, 4):
            if len(curve["x"]) < known + 2:
                continue
            for mask_index in range(args.masks):
                rng = stable_rng(curve["id"], known, mask_index, args.seed)
                obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
                case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
                draws = ensemble(model, case, checkpoint, device, args.steps, args.eta, args.ensemble)
                for target in TARGETS:
                    ref = monotone_decreasing(curve["gy"])
                    if not (ref.min() <= target <= ref.max()):
                        continue
                    true_x = invert_log_stress(curve["gx"], ref, target)
                    sample_x = np.asarray([invert_log_stress(curve["gx"], draw, target) for draw in draws])
                    lo_x, hi_x = np.quantile(sample_x, [0.05, 0.95])
                    records.append({
                        "split": split_name,
                        "curve_id": curve["id"],
                        "dataset": dataset_name(curve["id"]),
                        "known": known,
                        "mask": mask_index,
                        "target_logN": target,
                        "true_logS": true_x,
                        "pred_logS": float(sample_x.mean()),
                        "raw_lo_logS": float(lo_x),
                        "raw_hi_logS": float(hi_x),
                    })
    return records


def conformal_quantile(values: np.ndarray, alpha: float = 0.1) -> float:
    if not len(values):
        return 0.0
    level = min(1.0, math.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(x), kind="stable"), kind="stable")
    ry = np.argsort(np.argsort(np.asarray(y), kind="stable"), kind="stable")
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_accuracy(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    if len(true) < 2:
        return float("nan")
    i, j = np.triu_indices(len(true), 1)
    keep = np.abs(true[i] - true[j]) > 1e-10
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.sign(true[i][keep] - true[j][keep]) ==
                         np.sign(pred[i][keep] - pred[j][keep])))


def summarize(validation, test):
    calibration = {}
    for known in (2, 3, 4):
        for target in TARGETS:
            rows = [r for r in validation if r["known"] == known and r["target_logN"] == target]
            scores = [max(r["raw_lo_logS"] - r["true_logS"],
                          r["true_logS"] - r["raw_hi_logS"], 0.0) for r in rows]
            calibration[(known, target)] = conformal_quantile(np.asarray(scores))

    detail, summary = [], []
    for row in test:
        q = calibration[(row["known"], row["target_logN"])]
        item = dict(row)
        item["calibration_q_logS"] = q
        item["lo_logS"] = row["raw_lo_logS"] - q
        item["hi_logS"] = row["raw_hi_logS"] + q
        for key in ("true", "pred", "lo", "hi"):
            item[f"{key}_stress_mpa"] = 10 ** item[f"{key}_logS"]
        detail.append(item)

    for known in (2, 3, 4):
        for target in TARGETS:
            rows = [r for r in detail if r["known"] == known and r["target_logN"] == target]
            if not rows:
                continue
            true = np.asarray([r["true_stress_mpa"] for r in rows])
            pred = np.asarray([r["pred_stress_mpa"] for r in rows])
            lo = np.asarray([r["lo_stress_mpa"] for r in rows])
            hi = np.asarray([r["hi_stress_mpa"] for r in rows])
            summary.append({
                "known": known,
                "target_cycles": int(10 ** target),
                "cases": len(rows),
                "stress_mae_mpa": float(np.mean(np.abs(pred - true))),
                "relative_stress_error": float(np.mean(np.abs(pred - true) / np.maximum(true, 1e-9))),
                "spearman_rho": spearman(true, pred),
                "coverage90": float(np.mean((true >= lo) & (true <= hi))),
                "interval_width_mpa": float(np.mean(hi - lo)),
                "pairwise_ranking_accuracy": pairwise_accuracy(true, pred),
                "calibration_q_logS": calibration[(known, target)],
            })
    return calibration, detail, summary


def main(args):
    device, rank, _, world = setup_device(); args.world = world
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split_seed = checkpoint.get("split_seed", checkpoint.get("seed", args.split_seed))
    train, validation, test, _ = split_curves(args.data, split_seed,
                                               split_ids=checkpoint.get("split_ids"))
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_records = evaluate_split(validation, model, checkpoint, device, args, rank, "validation")
    test_records = evaluate_split(test, model, checkpoint, device, args, rank, "test")
    part = args.output_dir / f"fatigue_strength_rank{rank}.json"
    part.write_text(json.dumps({"validation": validation_records, "test": test_records}))
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_validation, all_test = [], []
        for r in range(world):
            payload = json.loads((args.output_dir / f"fatigue_strength_rank{r}.json").read_text())
            all_validation.extend(payload["validation"]); all_test.extend(payload["test"])
        calibration, detail, summary = summarize(all_validation, all_test)
        output = {
            "method": "PA-CBB",
            "checkpoint": str(args.checkpoint),
            "split_seed": split_seed,
            "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "steps": args.steps, "ensemble": args.ensemble, "eta": args.eta,
            "masks_per_curve": args.masks,
            "interval": "split-conformal expansion of raw 90% posterior interval in log10 stress",
            "calibration": {f"k{k}_N{int(10**t)}": q for (k, t), q in calibration.items()},
            "summary": summary,
        }
        (args.output_dir / "fatigue_strength_summary.json").write_text(json.dumps(output, indent=2))
        with (args.output_dir / "fatigue_strength_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        with (args.output_dir / "fatigue_strength_cases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
        print(json.dumps(output, indent=2), flush=True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--ensemble", type=int, default=64)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--masks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
