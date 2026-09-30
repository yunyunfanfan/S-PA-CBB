#!/usr/bin/env python3
"""Export aligned validation/test PA-CBB curve ensembles for downstream screening."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from control_bridge import BaseUNet, ControlBridge, generate, make_case
from control_score_bridge import split_curves
from deep_bridge import cleanup, setup_device


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31 - 1)


@torch.no_grad()
def export_split(curves, split_name, model, checkpoint, device, args, rank, world):
    rows = []
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
                tensors.append(value.repeat(args.ensemble, *([1] * (value.ndim - 1))))
            seed = stable_seed(curve["id"], known, split_name, args.draw_seed)
            torch.manual_seed(seed)
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.manual_seed_all(seed)
            draws = generate(model, *tensors, args.steps, args.eta).cpu().numpy()
            draws = draws * checkpoint["ystd"] + checkpoint["ymean"]
            rows.append({
                "key": (split_name, curve["id"], known),
                "gx": np.asarray(curve["gx"], np.float32),
                "truth": np.asarray(curve["gy"], np.float32),
                "observed": np.asarray(obs, np.int16),
                "draws": np.asarray(draws, np.float32),
            })
    return rows


def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split_seed = checkpoint.get("split_seed", checkpoint.get("seed", args.split_seed))
    _, validation, test, _ = split_curves(args.data, split_seed,
                                           split_ids=checkpoint.get("split_ids"))
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = export_split(validation, "validation", model, checkpoint, device, args, rank, world)
    rows += export_split(test, "test", model, checkpoint, device, args, rank, world)
    payload = {
        "method": "PA-CBB", "steps": args.steps, "ensemble": args.ensemble,
        "eta": args.eta, "mask_seed": args.mask_seed, "draw_seed": args.draw_seed,
        "split_seed": split_seed, "rows": rows,
    }
    with gzip.open(args.output_dir / f"pacbb_rank{rank}.pkl.gz", "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if world > 1:
        dist.barrier()
    if rank == 0:
        (args.output_dir / "PACBB_COMPLETE").touch()
        print({"validation": len(validation), "test": len(test), "world": world})
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
    parser.add_argument("--draw-seed", type=int, default=20261110)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
