#!/usr/bin/env python3
"""Reverse-step and posterior-ensemble-size sweeps for a fixed full model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ablation_control_variants import sample
from control_bridge import BaseUNet, ControlBridge, make_case
from control_score_bridge import split_curves
from deep_bridge import cleanup, setup_device
from extended_metrics import curve_energy, empirical_crps


STEPS = (1, 3, 5, 9, 16)
ENSEMBLES = (4, 8, 16, 32, 64)


def case_seed(curve_id, known, seed):
    token = f"{curve_id}|{known}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def metric_record(curve, case, draws, elapsed):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), case["obs"])
    values = np.asarray([[np.interp(curve["x"][i], curve["gx"], draw)
                          for i in hidden] for draw in draws])
    truth = curve["y"][hidden]; mean = values.mean(axis=0)
    low, high = np.quantile(values, [0.05, 0.95], axis=0)
    return {
        "mae_logN": float(np.mean(np.abs(mean - truth))),
        "crps_logN": float(np.mean(empirical_crps(values, truth))),
        "energy_score": float(curve_energy(draws, curve["gy"])),
        "coverage90": float(np.mean((truth >= low) & (truth <= high))),
        "interval_width_logN": float(np.mean(high - low)),
        "seconds_per_curve": float(elapsed),
    }


@torch.no_grad()
def run(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, curves, _ = split_curves(args.data, checkpoint["split_seed"],
                                    split_ids=checkpoint["split_ids"])
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    records = []
    # Compile/warm the device kernels before latency measurements.
    warm_curve = curves[rank::world][0]
    warm_rng = np.random.default_rng(case_seed(warm_curve["id"], 2, args.mask_seed))
    warm_obs = np.sort(warm_rng.choice(len(warm_curve["x"]), 2, replace=False))
    warm_case = make_case(warm_curve, warm_obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
    warm_tensors = []
    for name in ("static", "control", "raw", "phys", "mask", "oy"):
        value = torch.tensor(warm_case[name], device=device).unsqueeze(0)
        warm_tensors.append(value.repeat(4, *([1] * (value.ndim - 1))))
    sample(model, "full", *warm_tensors, 2, args.eta)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    for curve in curves[rank::world]:
        for known in (2, 3, 4):
            if len(curve["x"]) < known + 2:
                continue
            rng = np.random.default_rng(case_seed(curve["id"], known, args.mask_seed))
            obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
            case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
            tensors = []
            for name in ("static", "control", "raw", "phys", "mask", "oy"):
                value = torch.tensor(case[name], device=device).unsqueeze(0)
                tensors.append(value.repeat(64, *([1] * (value.ndim - 1))))
            for reverse_steps in STEPS:
                start = time.perf_counter()
                normalized = sample(model, "full", *[v[:args.step_ensemble] for v in tensors],
                                    reverse_steps, args.eta)
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.synchronize()
                elapsed = (time.perf_counter() - start)
                draws = normalized.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]
                record = metric_record(curve, case, draws, elapsed)
                record.update({"sweep": "steps", "value": reverse_steps,
                               "known": known, "curve_id": curve["id"]})
                records.append(record)
            start = time.perf_counter()
            normalized = sample(model, "full", *tensors, args.ensemble_steps, args.eta)
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            full_draws = normalized.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]
            for size in ENSEMBLES:
                record = metric_record(curve, case, full_draws[:size], elapsed * size / 64)
                record.update({"sweep": "ensemble", "value": size,
                               "known": known, "curve_id": curve["id"]})
                records.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"sampling_sweep_rank{rank}.json").write_text(json.dumps(records))
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for r in range(world):
            merged.extend(json.loads((args.output_dir / f"sampling_sweep_rank{r}.json").read_text()))
        summary = []
        for sweep, values in (("steps", STEPS), ("ensemble", ENSEMBLES)):
            for value in values:
                rows = [row for row in merged if row["sweep"] == sweep and row["value"] == value]
                item = {"sweep": sweep, "value": value, "curves": len(rows)}
                for metric in ("mae_logN", "crps_logN", "energy_score", "coverage90",
                               "interval_width_logN", "seconds_per_curve"):
                    item[metric] = float(np.mean([row[metric] for row in rows]))
                summary.append(item)
        payload = {"checkpoint": str(args.checkpoint), "eta": args.eta,
                   "step_sweep_ensemble": args.step_ensemble,
                   "ensemble_sweep_steps": args.ensemble_steps,
                   "summary": summary, "case_records": merged}
        (args.output_dir / "sampling_sweeps.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({k: v for k, v in payload.items() if k != "case_records"}, indent=2))
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-ensemble", type=int, default=32)
    parser.add_argument("--ensemble-steps", type=int, default=9)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    run(parser.parse_args())
