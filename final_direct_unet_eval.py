#!/usr/bin/env python3
"""Direct U-Net evaluation on the final hashed sparse masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baseline_benchmark import curve_metrics, point_metrics
from control_score_bridge import split_curves
from deep_bridge import cleanup, setup_device
from direct_unet import DirectUNet, make_direct_case


def mask_for(curve, known, seed):
    token = f'{curve["id"]}|{known}|{seed}'.encode(); value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.sort(np.random.default_rng(value).choice(len(curve["x"]), known, replace=False))


@torch.no_grad()
def main(args):
    device, rank, _, world = setup_device()
    if world != 1 or rank != 0: raise RuntimeError("Direct U-Net final evaluation uses one device")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, test, _ = split_curves(args.data, checkpoint["split_seed"], split_ids=checkpoint["split_ids"])
    model = DirectUNet(checkpoint["base"]).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    output = {"method": "Direct U-Net", "mask_seed": args.mask_seed, "known_points": {}}
    for known in (2, 3, 4):
        cases = [make_direct_case(curve, mask_for(curve, known, args.mask_seed), checkpoint["ymean"], checkpoint["ystd"])
                 for curve in test if len(curve["x"]) >= known + 2]
        grids = []
        for start in range(0, len(cases), args.batch):
            batch = cases[start:start + args.batch]
            channels = torch.tensor(np.stack([row["channels"] for row in batch]), device=device)
            raw = torch.tensor(np.stack([row["raw"] for row in batch]), device=device)
            mask = torch.tensor(np.stack([row["mask"] for row in batch]), device=device)
            values = torch.tensor(np.stack([row["values"] for row in batch]), device=device)
            pred = model(channels, raw) * (1 - mask) + values * mask
            grids.extend(pred.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"])
        truth, pred = [], []
        for row, grid in zip(cases, grids):
            curve, obs = row["curve"], row["obs"]; hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
            truth.extend(curve["y"][hidden]); pred.extend(np.interp(curve["x"][hidden], curve["gx"], grid))
        output["known_points"][str(known)] = {"point": point_metrics(truth, pred),
                                               "curve": curve_metrics([row["curve"] for row in cases], grids)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2)); print(json.dumps(output, indent=2)); cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mask-seed", type=int, default=20261012); parser.add_argument("--batch", type=int, default=64); main(parser.parse_args())
