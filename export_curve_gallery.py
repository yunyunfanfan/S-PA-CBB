#!/usr/bin/env python3
"""Export representative full posterior S-N curves for publication figures."""
from __future__ import annotations

import argparse
import json
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


def serial(values):
    return np.asarray(values).astype(float).tolist()


@torch.no_grad()
def main(args):
    device, rank, _, world = setup_device()
    if world != 1 or rank != 0:
        raise RuntimeError("This compact gallery exporter is intended for a single device.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, curves, _ = split_curves(
        args.data, ck["seed"], split_ids=ck.get("split_ids")
    )
    model = ScoreControl(ScoreBase(ck["base"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    rng = np.random.default_rng(args.seed)

    candidates = {k: [] for k in (2, 3, 4)}
    for k in (2, 3, 4):
        cases = []
        for curve in curves:
            if len(curve["x"]) < k + 2:
                continue
            obs = np.sort(rng.choice(len(curve["x"]), k, replace=False))
            case = make_case(
                curve, obs, ck["prior"], ck["ymean"], ck["ystd"]
            )
            case["curve"] = curve
            cases.append(case)

        for start in range(0, len(cases), args.batch):
            batch = cases[start : start + args.batch]
            tensors = [
                torch.tensor(np.stack([q[name] for q in batch]), device=device)
                for name in ("static", "control", "raw", "phys", "mask", "oy")
            ]
            static, control, raw, phys, mask, oy = tensors
            draws = np.stack(
                [
                    sample(
                        model,
                        static,
                        control,
                        raw,
                        phys,
                        mask,
                        oy,
                        args.steps,
                        ck["sigma"],
                        args.eta,
                    )
                    .cpu()
                    .numpy()
                    for _ in range(args.ensemble)
                ]
            )
            for j, case in enumerate(batch):
                posterior = draws[:, j] * ck["ystd"] + ck["ymean"]
                mean = posterior.mean(axis=0)
                hidden = np.setdiff1d(
                    np.arange(len(case["y"])), case["obs"]
                )
                truth_hidden = case["y"][hidden]
                mean_hidden = np.interp(
                    case["x"][hidden], case["gx"], mean
                )
                error = float(np.mean(np.abs(mean_hidden - truth_hidden)))
                candidates[k].append((error, case, posterior))

    records = []
    quantiles = (
        np.linspace(0.0, min(0.08, 0.04 * max(args.per_k - 1, 0)), args.per_k)
        if args.selection == "best"
        else np.linspace(0.12, 0.88, args.per_k)
    )
    for k in (2, 3, 4):
        pool = sorted(candidates[k], key=lambda item: item[0])
        used_ids = set()
        used_names = set()
        for quantile in quantiles:
            target = int(round(quantile * (len(pool) - 1)))
            order = sorted(range(len(pool)), key=lambda idx: abs(idx - target))
            chosen = None
            for idx in order:
                curve = pool[idx][1]["curve"]
                if curve["id"] not in used_ids and curve["name"] not in used_names:
                    chosen = idx
                    break
            if chosen is None:
                for idx in order:
                    if pool[idx][1]["curve"]["id"] not in used_ids:
                        chosen = idx
                        break
            error, case, posterior = pool[chosen]
            curve = case["curve"]
            used_ids.add(curve["id"])
            used_names.add(curve["name"])
            records.append(
                {
                    "k": k,
                    "error_mae_logN": error,
                    "id": curve["id"],
                    "name": curve["name"],
                    "family": curve["family"],
                    "am_type": curve["am"],
                    "load_ratio": curve["R"],
                    "test_type": curve["test"],
                    "gx_log_stress": serial(case["gx"]),
                    "truth_grid_logN": serial(curve["gy"]),
                    "basquin_grid_logN": serial(
                        case["phys"] * ck["ystd"] + ck["ymean"]
                    ),
                    "posterior_logN": serial(posterior),
                    "raw_log_stress": serial(case["x"]),
                    "raw_logN": serial(case["y"]),
                    "observed_indices": np.asarray(case["obs"]).astype(int).tolist(),
                }
            )

    payload = {
        "model": "physics-anchored ControlNet Brownian bridge",
        "split": "source-paper holdout",
        "steps": args.steps,
        "eta": args.eta,
        "ensemble": args.ensemble,
        "selection": (
            "lowest-error cases within each k, preferring distinct materials"
            if args.selection == "best"
            else "within-k error quantiles, preferring distinct materials"
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(args.output, flush=True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--per-k", type=int, default=4)
    parser.add_argument("--selection", choices=("quantiles", "best"), default="quantiles")
    parser.add_argument("--seed", type=int, default=20261002)
    main(parser.parse_args())
