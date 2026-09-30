#!/usr/bin/env python3
"""Export same-mask curve cases where the complete method beats all main baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from baseline_benchmark import build_training, feature, traditional
from control_bridge import make_case
from control_score_bridge import ScoreBase, ScoreControl, sample, split_curves
from deep_bridge import cleanup, fit_line, setup_device
from direct_unet import DirectUNet, make_direct_case


def serial(values):
    return np.asarray(values).astype(float).tolist()


def hidden_mae(curve, obs, prediction):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
    estimate = np.interp(curve["x"][hidden], curve["gx"], prediction)
    return float(np.mean(np.abs(estimate - curve["y"][hidden])))


@torch.no_grad()
def main(args):
    device, rank, _, world = setup_device()
    if world != 1 or rank != 0:
        raise RuntimeError("Advantage-gallery export uses one accelerator.")

    torch.manual_seed(args.mask_seed)
    np.random.seed(args.mask_seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train, _, curves, _ = split_curves(args.data, ck["seed"], split_ids=ck.get("split_ids"))
    model = ScoreControl(ScoreBase(ck["base"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    direct_ck = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    direct = DirectUNet(direct_ck["base"]).to(device)
    direct.load_state_dict(direct_ck["model"])
    direct.eval()

    slopes, yvals = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope)
        yvals.extend(curve["y"])
    prior = float(np.median(slopes))
    ymean, ystd = float(np.mean(yvals)), float(np.std(yvals))
    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.baseline_seed + 11)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=8.0))
    ridge.fit(xtrain, ytrain)

    rng = np.random.default_rng(args.mask_seed)
    candidates = {k: [] for k in (2, 3, 4)}
    for k in (2, 3, 4):
        cases = []
        for curve in curves:
            if len(curve["x"]) < k + 2:
                continue
            obs = np.sort(rng.choice(len(curve["x"]), k, replace=False))
            case = make_case(curve, obs, ck["prior"], ck["ymean"], ck["ystd"])
            case["curve"] = curve
            cases.append(case)

        for start in range(0, len(cases), args.batch):
            batch = cases[start : start + args.batch]
            tensors = [
                torch.tensor(np.stack([q[name] for q in batch]), device=device)
                for name in ("static", "control", "raw", "phys", "mask", "oy")
            ]
            static, control, raw, phys, mask, oy = tensors
            draws = np.stack([
                sample(model, static, control, raw, phys, mask, oy, args.steps, ck["sigma"], args.eta).cpu().numpy()
                for _ in range(args.ensemble)
            ])

            direct_cases = [make_direct_case(q["curve"], q["obs"], direct_ck["ymean"], direct_ck["ystd"]) for q in batch]
            dchannels = torch.tensor(np.stack([q["channels"] for q in direct_cases]), device=device)
            draw = torch.tensor(np.stack([q["raw"] for q in direct_cases]), device=device)
            dmask = torch.tensor(np.stack([q["mask"] for q in direct_cases]), device=device)
            dvalues = torch.tensor(np.stack([q["values"] for q in direct_cases]), device=device)
            direct_grids = (direct(dchannels, draw) * (1 - dmask) + dvalues).cpu().numpy()
            direct_grids = np.clip(direct_grids * direct_ck["ystd"] + direct_ck["ymean"], 2, 10)

            for j, case in enumerate(batch):
                curve, obs = case["curve"], case["obs"]
                posterior = draws[:, j] * ck["ystd"] + ck["ymean"]
                proposed = posterior.mean(axis=0)
                pchip = traditional(curve, obs, prior)["PCHIP"]
                ridge_grid = np.clip(ridge.predict(feature(curve, obs, ymean, ystd)[None])[0] * ystd + ymean, 2, 10)
                comparison = {"PCHIP": pchip, "Ridge": ridge_grid, "Direct U-Net": direct_grids[j]}
                proposed_mae = hidden_mae(curve, obs, proposed)
                comparison_mae = {name: hidden_mae(curve, obs, grid) for name, grid in comparison.items()}
                advantage = min(comparison_mae.values()) - proposed_mae
                if advantage <= args.minimum_margin:
                    continue
                candidates[k].append({
                    "k": k,
                    "advantage_margin_logN": float(advantage),
                    "error_mae_logN": proposed_mae,
                    "id": curve["id"], "name": curve["name"], "family": curve["family"],
                    "am_type": curve["am"], "load_ratio": curve["R"], "test_type": curve["test"],
                    "gx_log_stress": serial(case["gx"]), "truth_grid_logN": serial(curve["gy"]),
                    "posterior_logN": serial(posterior), "raw_log_stress": serial(case["x"]),
                    "raw_logN": serial(case["y"]), "observed_indices": np.asarray(obs).astype(int).tolist(),
                    "comparison_grid_logN": {name: serial(grid) for name, grid in comparison.items()},
                    "comparison_hidden_mae_logN": comparison_mae,
                })

    records = []
    counts = {}
    for k in (2, 3, 4):
        pool = sorted(candidates[k], key=lambda q: q["advantage_margin_logN"], reverse=True)
        counts[str(k)] = len(pool)
        chosen, used_names, used_families = [], set(), set()
        for record in pool:
            if record["name"] in used_names or record["family"] in used_families:
                continue
            chosen.append(record); used_names.add(record["name"]); used_families.add(record["family"])
            if len(chosen) == args.per_k: break
        if len(chosen) < args.per_k:
            for record in pool:
                if record["id"] in {q["id"] for q in chosen}: continue
                chosen.append(record)
                if len(chosen) == args.per_k: break
        if len(chosen) < args.per_k:
            raise RuntimeError(f"Only {len(chosen)} qualifying cases for k={k}; reduce --minimum-margin.")
        records.extend(chosen)

    payload = {
        "model": "physics-anchored ControlNet Brownian bridge",
        "split": "source-paper holdout",
        "steps": args.steps, "eta": args.eta, "ensemble": args.ensemble,
        "selection": "largest same-mask MAE advantage over PCHIP, Ridge and Direct U-Net, preferring distinct material families",
        "minimum_margin_logN": args.minimum_margin,
        "qualifying_cases_by_k": counts,
        "comparison_methods": ["PCHIP", "Ridge", "Direct U-Net"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), "qualifying_cases_by_k": counts}, indent=2), flush=True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--per-k", type=int, default=4)
    parser.add_argument("--minimum-margin", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--baseline-seed", type=int, default=20261003)
    parser.add_argument("--mask-seed", type=int, default=20261002)
    main(parser.parse_args())
