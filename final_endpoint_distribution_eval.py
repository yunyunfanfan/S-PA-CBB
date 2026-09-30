#!/usr/bin/env python3
"""Final aligned 3-step endpoint-sampler evaluation with conformal calibration."""
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

from ablation_control_variants import sample
from control_bridge import BaseUNet, ControlBridge, make_case
from control_score_bridge import split_curves
from deep_bridge import cleanup, setup_device
from extended_metrics import curve_energy, empirical_crps, interval_score, summarize_point


def mask_for(curve, known, seed):
    token = f'{curve["id"]}|{known}|{seed}'.encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.sort(np.random.default_rng(value).choice(len(curve["x"]), known, replace=False))


@torch.no_grad()
def evaluate_split(model, curves, checkpoint, device, args, rank, split_name):
    records = []
    for curve in curves[rank::dist.get_world_size() if dist.is_initialized() else 1]:
        for known in (2, 3, 4):
            if len(curve["x"]) < known + 2:
                continue
            obs = mask_for(curve, known, args.mask_seed)
            case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
            tensors = []
            for name in ("static", "control", "raw", "phys", "mask", "oy"):
                value = torch.tensor(case[name], device=device).unsqueeze(0)
                tensors.append(value.repeat(args.ensemble, *([1] * (value.ndim - 1))))
            torch.manual_seed(int.from_bytes(hashlib.sha256(f'{split_name}|{curve["id"]}|{known}|{args.seed}|{rank}'.encode()).digest()[:4], "little"))
            tic = time.perf_counter()
            normalized = sample(model, checkpoint["variant"], *tensors, args.steps, args.eta)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            elapsed = time.perf_counter() - tic
            draws = normalized.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]
            hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
            values = np.stack([np.interp(curve["x"][hidden], curve["gx"], draw) for draw in draws])
            truth = curve["y"][hidden]; pred = values.mean(0)
            lo, hi = np.quantile(values, [0.05, 0.95], axis=0)
            mean_grid = draws.mean(0)
            records.append({
                "split": split_name, "curve_id": curve["id"], "known": known,
                "truth": truth.tolist(), "prediction": pred.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                "grid_mae": float(np.mean(np.abs(mean_grid - curve["gy"]))),
                "slope_error": float(abs(np.polyfit(curve["gx"], mean_grid, 1)[0] - np.polyfit(curve["gx"], curve["gy"], 1)[0])),
                "monotonic_violation": float(np.mean(np.diff(draws, axis=1) > 0)),
                "crps": float(np.mean(empirical_crps(values, truth))),
                "energy": float(curve_energy(draws, curve["gy"])),
                "seconds_per_curve": elapsed,
            })
    return records


def summarize(validation, test):
    output = {}
    for known in (2, 3, 4):
        val = [row for row in validation if row["known"] == known]
        rows = [row for row in test if row["known"] == known]
        vt = np.concatenate([np.asarray(row["truth"]) for row in val]); vl = np.concatenate([np.asarray(row["lo"]) for row in val]); vh = np.concatenate([np.asarray(row["hi"]) for row in val])
        scores = np.maximum(vl - vt, vt - vh)
        q_level = min(1.0, math.ceil((len(scores) + 1) * 0.90) / len(scores))
        q = float(np.quantile(scores, q_level, method="higher"))
        truth = np.concatenate([np.asarray(row["truth"]) for row in rows]); pred = np.concatenate([np.asarray(row["prediction"]) for row in rows]); lo = np.concatenate([np.asarray(row["lo"]) for row in rows]); hi = np.concatenate([np.asarray(row["hi"]) for row in rows])
        output[str(known)] = {
            "point": summarize_point(truth, pred),
            "grid_mae_logN": float(np.mean([row["grid_mae"] for row in rows])),
            "slope_error": float(np.mean([row["slope_error"] for row in rows])),
            "monotonic_violation": float(np.mean([row["monotonic_violation"] for row in rows])),
            "crps_logN": float(np.mean([row["crps"] for row in rows])),
            "energy_score": float(np.mean([row["energy"] for row in rows])),
            "raw_coverage90": float(np.mean((truth >= lo) & (truth <= hi))),
            "raw_width90_logN": float(np.mean(hi - lo)),
            "raw_interval_score90": float(np.mean(interval_score(truth, lo, hi, 0.1))),
            "conformal_q": q,
            "conformal_coverage90": float(np.mean((truth >= lo - q) & (truth <= hi + q))),
            "conformal_width90_logN": float(np.mean(hi - lo + 2 * q)),
            "seconds_per_curve": float(np.mean([row["seconds_per_curve"] for row in rows])),
            "curves": len(rows),
        }
    return output


def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, validation, test, _ = split_curves(args.data, checkpoint["split_seed"], split_ids=checkpoint["split_ids"])
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    validation_rows = evaluate_split(model, validation, checkpoint, device, args, rank, "validation")
    test_rows = evaluate_split(model, test, checkpoint, device, args, rank, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps(validation_rows + test_rows))
    if world > 1:
        dist.barrier()
    if rank == 0:
        records = []
        for index in range(world):
            records.extend(json.loads((args.output_dir / f"rank{index}.json").read_text()))
        validation_all = [row for row in records if row["split"] == "validation"]
        test_all = [row for row in records if row["split"] == "test"]
        result = {"sampler": "exact physical endpoint plus intermediate Gaussian transitions",
                  "steps": args.steps, "ensemble": args.ensemble, "eta": args.eta,
                  "mask_seed": args.mask_seed, "summary": summarize(validation_all, test_all),
                  "case_records": records}
        (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "case_records"}, indent=2))
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
    parser.add_argument("--seed", type=int, default=20261030)
    main(parser.parse_args())
