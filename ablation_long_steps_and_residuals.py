#!/usr/bin/env python3
"""Noise, long-step, ensemble-value, and physics-residual audits for PA-CBB.

The long-step sweep deliberately uses one posterior sample.  Ensemble and noise
audits are separate so that solver depth is not confounded with Monte Carlo
averaging.  The sampler is the deployed endpoint sampler: t=1 is the physical
endpoint, and stochasticity is injected on intermediate reverse transitions.
Consequently, a one-step endpoint-to-data map is deterministic by construction.
"""
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
from extended_metrics import curve_energy, empirical_crps


LONG_STEPS = (1, 2, 3, 5, 9, 16, 32, 64, 128)
ENSEMBLES = (1, 2, 4, 8, 16, 32, 64)
ENSEMBLE_STEPS = (3, 9)
ETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def stable_seed(*parts):
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31)


def synchronize():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def tensors_for(cases, device, repeats=1):
    result = []
    for name in ("static", "control", "raw", "phys", "mask", "oy"):
        value = torch.tensor(np.stack([case[name] for _, case in cases]), device=device)
        if repeats > 1:
            value = value[:, None].repeat(1, repeats, *([1] * (value.ndim - 1)))
            value = value.reshape(len(cases) * repeats, *value.shape[2:])
        result.append(value)
    return result


def one_draw_metrics(curve, case, draw, elapsed):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), case["obs"])
    hidden_pred = np.array([np.interp(curve["x"][i], curve["gx"], draw) for i in hidden])
    slope_pred = np.polyfit(curve["gx"], draw, 1)[0]
    slope_true = np.polyfit(curve["gx"], curve["gy"], 1)[0]
    return {
        "hidden_mae": float(np.mean(np.abs(hidden_pred - curve["y"][hidden]))),
        "grid_mae": float(np.mean(np.abs(draw - curve["gy"]))),
        "energy_score": float(curve_energy(draw[None], curve["gy"])),
        "slope_error": float(abs(slope_pred - slope_true)),
        "violation_rate": float(np.mean(np.diff(draw) > 0)),
        "seconds_per_curve": float(elapsed),
    }


def ensemble_metrics(curve, case, draws, elapsed):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), case["obs"])
    values = np.asarray([[np.interp(curve["x"][i], curve["gx"], draw)
                          for i in hidden] for draw in draws])
    truth = curve["y"][hidden]
    mean = draws.mean(axis=0)
    low, high = np.quantile(values, [0.05, 0.95], axis=0)
    grid_abs = np.abs(draws - curve["gy"][None])
    centered = draws - mean[None]
    diversity = float(np.sqrt(np.mean(centered ** 2)))
    return {
        "hidden_mae": float(np.mean(np.abs(values.mean(0) - truth))),
        "grid_mae": float(np.mean(np.abs(mean - curve["gy"]))),
        "crps": float(np.mean(empirical_crps(values, truth))),
        "energy_score": float(curve_energy(draws, curve["gy"])),
        "coverage90": float(np.mean((truth >= low) & (truth <= high))),
        "width90": float(np.mean(high - low)),
        "best_of_m_grid_mae": float(np.min(np.mean(grid_abs, axis=1))),
        "diversity_rms": diversity,
        "spread_mean": float(np.mean(np.std(draws, axis=0))),
        "seconds_per_curve": float(elapsed),
    }


def residual_record(curve, case, draws):
    physics = case["phys"] * case["ystd"] + case["ymean"] if "ystd" in case else None
    if physics is None:
        raise RuntimeError("case normalization metadata missing")
    mean = draws.mean(axis=0); truth = curve["gy"]
    required = truth - physics; learned = mean - physics
    base_residual = physics - truth; generated_residual = mean - truth
    return {
        "curve_id": curve["id"],
        "known": int(len(case["obs"])),
        "physics_mae": float(np.mean(np.abs(base_residual))),
        "generated_mae": float(np.mean(np.abs(generated_residual))),
        "improved": bool(np.mean(np.abs(generated_residual)) < np.mean(np.abs(base_residual))),
        "physics_slope_error": float(abs(np.polyfit(curve["gx"], physics, 1)[0] - np.polyfit(curve["gx"], truth, 1)[0])),
        "generated_slope_error": float(abs(np.polyfit(curve["gx"], mean, 1)[0] - np.polyfit(curve["gx"], truth, 1)[0])),
        "mean_spread": float(np.mean(np.std(draws, axis=0))),
        "grid_position": np.linspace(0, 1, len(truth)).tolist(),
        "required_correction": required.tolist(),
        "learned_correction": learned.tolist(),
        "physics_residual": base_residual.tolist(),
        "generated_residual": generated_residual.tolist(),
    }


@torch.no_grad()
def run(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, test_curves, _ = split_curves(args.data, checkpoint["split_seed"],
                                         split_ids=checkpoint["split_ids"])
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()

    cases = []
    for curve in test_curves[rank::world]:
        for known in (2, 3, 4):
            if len(curve["x"]) < known + 2:
                continue
            rng = np.random.default_rng(stable_seed(curve["id"], known, args.mask_seed))
            obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
            case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
            case["ymean"], case["ystd"] = checkpoint["ymean"], checkpoint["ystd"]
            cases.append((curve, case))

    # Device warm-up is excluded from reported timings.
    warm = cases[:min(4, len(cases))]
    torch.manual_seed(stable_seed("warm", rank)); sample(model, "full", *tensors_for(warm, device), 3, args.eta)
    synchronize()

    long_records, ensemble_records, noise_records, residual_records = [], [], [], []
    for start in range(0, len(cases), args.batch):
        batch = cases[start:start + args.batch]
        base_tensors = tensors_for(batch, device)

        for steps in LONG_STEPS:
            torch.manual_seed(stable_seed("long", start, steps, rank, args.seed))
            tic = time.perf_counter(); normalized = sample(model, "full", *base_tensors, steps, args.eta)
            synchronize(); elapsed = (time.perf_counter() - tic) / len(batch)
            actual = normalized.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]
            for (curve, case), draw in zip(batch, actual):
                item = one_draw_metrics(curve, case, draw, elapsed)
                item.update({"curve_id": curve["id"], "known": len(case["obs"]), "steps": steps})
                long_records.append(item)

        for steps in ENSEMBLE_STEPS:
            tensors = tensors_for(batch, device, repeats=max(ENSEMBLES))
            torch.manual_seed(stable_seed("ensemble", start, steps, rank, args.seed))
            tic = time.perf_counter(); normalized = sample(model, "full", *tensors, steps, args.eta)
            synchronize(); elapsed = (time.perf_counter() - tic) / len(batch)
            actual = normalized.cpu().numpy().reshape(len(batch), max(ENSEMBLES), -1)
            actual = actual * checkpoint["ystd"] + checkpoint["ymean"]
            for (curve, case), all_draws in zip(batch, actual):
                for size in ENSEMBLES:
                    item = ensemble_metrics(curve, case, all_draws[:size], elapsed * size / max(ENSEMBLES))
                    item.update({"curve_id": curve["id"], "known": len(case["obs"]),
                                 "steps": steps, "ensemble": size})
                    ensemble_records.append(item)
                if steps == args.residual_steps:
                    residual_records.append(residual_record(curve, case, all_draws))

        for eta in ETAS:
            tensors = tensors_for(batch, device, repeats=args.noise_ensemble)
            torch.manual_seed(stable_seed("eta", start, eta, rank, args.seed))
            tic = time.perf_counter(); normalized = sample(model, "full", *tensors, args.noise_steps, eta)
            synchronize(); elapsed = (time.perf_counter() - tic) / len(batch)
            actual = normalized.cpu().numpy().reshape(len(batch), args.noise_ensemble, -1)
            actual = actual * checkpoint["ystd"] + checkpoint["ymean"]
            for (curve, case), draws in zip(batch, actual):
                item = ensemble_metrics(curve, case, draws, elapsed)
                item.update({"curve_id": curve["id"], "known": len(case["obs"]), "eta": eta,
                             "steps": args.noise_steps, "ensemble": args.noise_ensemble})
                noise_records.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_payload = {"long_steps": long_records, "ensembles": ensemble_records,
                    "noise": noise_records, "residuals": residual_records}
    (args.output_dir / f"audit_rank{rank}.json").write_text(json.dumps(rank_payload))
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {key: [] for key in rank_payload}
        for worker in range(world):
            payload = json.loads((args.output_dir / f"audit_rank{worker}.json").read_text())
            for key in merged:
                merged[key].extend(payload[key])

        def summarize(rows, keys, metrics):
            groups = {}
            for row in rows:
                label = tuple(row[key] for key in keys)
                groups.setdefault(label, []).append(row)
            result = []
            for label, values in sorted(groups.items()):
                item = dict(zip(keys, label)); item["curves"] = len(values)
                for metric in metrics:
                    data = np.asarray([value[metric] for value in values], float)
                    item[metric] = float(data.mean())
                    item[f"{metric}_se"] = float(data.std(ddof=1) / math.sqrt(len(data)))
                result.append(item)
            return result

        metric_names = ("hidden_mae", "grid_mae", "energy_score", "slope_error",
                        "violation_rate", "seconds_per_curve")
        ensemble_names = ("hidden_mae", "grid_mae", "crps", "energy_score", "coverage90",
                          "width90", "best_of_m_grid_mae", "diversity_rms", "spread_mean",
                          "seconds_per_curve")
        output = {
            "sampler": {"endpoint_start": True, "initial_state": "physical endpoint at t=1",
                        "transition_noise": "eta*sqrt(t_next*(1-t_next)/steps)*N(0,I)",
                        "one_step_is_deterministic": True},
            "settings": {"eta": args.eta, "long_steps": LONG_STEPS,
                         "ensemble_steps": ENSEMBLE_STEPS, "ensemble_sizes": ENSEMBLES,
                         "noise_etas": ETAS, "noise_steps": args.noise_steps,
                         "noise_ensemble": args.noise_ensemble},
            "long_step_summary": summarize(merged["long_steps"], ("steps",), metric_names),
            "long_step_by_k": summarize(merged["long_steps"], ("steps", "known"), metric_names),
            "ensemble_summary": summarize(merged["ensembles"], ("steps", "ensemble"), ensemble_names),
            "noise_summary": summarize(merged["noise"], ("eta",), ensemble_names),
            "residual_records": merged["residuals"],
        }
        (args.output_dir / "long_steps_noise_residuals.json").write_text(json.dumps(output, indent=2))
        print(json.dumps({key: value for key, value in output.items() if key != "residual_records"}, indent=2))
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--noise-steps", type=int, default=3)
    parser.add_argument("--noise-ensemble", type=int, default=32)
    parser.add_argument("--residual-steps", type=int, default=3)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    parser.add_argument("--seed", type=int, default=20261029)
    run(parser.parse_args())
