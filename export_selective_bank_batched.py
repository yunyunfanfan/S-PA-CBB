#!/usr/bin/env python3
"""Export validation/test posterior banks for selective S-PA-CBB evaluation.

Unlike the original downstream exporter, this implementation batches several
curves and all posterior draws in one model call.  It preserves the checkpoint
split IDs, so a LODO target is never used to fit the gate or calibration layer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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


def aligned_mask_rng(curve_id: str, known: int, seed: int):
    """Match external_baseline_benchmark mask index zero exactly."""
    token = f"{curve_id}|{known}|0|{seed}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.random.default_rng(value)


@torch.no_grad()
def export_split(curves, split_name, model, checkpoint, device, args, rank, world):
    local = curves[rank::world]
    rows = []
    for known in (2, 3, 4):
        cases = []
        for curve in local:
            if len(curve["x"]) < known + 2:
                continue
            rng = aligned_mask_rng(curve["id"], known, args.mask_seed)
            obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
            cases.append((curve, obs, make_case(
                curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])))

        for start in range(0, len(cases), args.case_batch):
            batch = cases[start:start + args.case_batch]
            names = ("static", "control", "raw", "phys", "mask", "oy")
            tensors = [torch.tensor(np.stack([item[2][name] for item in batch]), device=device)
                       for name in names]
            expanded = [value.repeat_interleave(args.ensemble, dim=0) for value in tensors]
            seed = stable_seed(split_name, known, start, args.draw_seed)
            torch.manual_seed(seed)
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.manual_seed_all(seed)
            generated = generate(model, *expanded, args.steps, args.eta)
            generated = generated.reshape(len(batch), args.ensemble, -1).cpu().numpy()
            generated = generated * checkpoint["ystd"] + checkpoint["ymean"]
            for index, (curve, obs, _case) in enumerate(batch):
                rows.append({
                    "key": (split_name, curve["id"], known),
                    "gx": np.asarray(curve["gx"], np.float32),
                    "truth": np.asarray(curve["gy"], np.float32),
                    "observed": np.asarray(obs, np.int16),
                    "draws": np.asarray(generated[index], np.float32),
                })
        if rank == 0:
            print(json.dumps({"split": split_name, "known": known,
                              "local_cases": len(cases)}), flush=True)
    return rows


def main(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split_seed = checkpoint.get("split_seed", checkpoint.get("seed", args.split_seed))
    _, validation, test, _ = split_curves(
        args.data, split_seed, split_ids=checkpoint.get("split_ids"))
    model = ControlBridge(BaseUNet(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = export_split(validation, "validation", model, checkpoint, device,
                        args, rank, world)
    rows += export_split(test, "test", model, checkpoint, device,
                         args, rank, world)
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        print(json.dumps({"validation_curves": len(validation), "test_curves": len(test),
                          "world": world, "device": str(device)}), flush=True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=20261030)
    parser.add_argument("--draw-seed", type=int, default=20261110)
    parser.add_argument("--split-seed", type=int, default=20261031)
    parser.add_argument("--case-batch", type=int, default=24)
    main(parser.parse_args())
