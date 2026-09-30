#!/usr/bin/env python3
"""Conditional Brownian bridge ablation with no Basquin/physics intermediate state."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from control_score_bridge import ScoreBase, ScoreControl, bridge_loss, epochs, sample, split_curves
from deep_bridge import GRID, hvec, setup_device, cleanup
from extended_metrics import empirical_crps, curve_energy
from direct_unet import point_metrics


def make_case(curve, obs, ymean, ystd):
    target = (curve["gy"] - ymean) / ystd
    prior = np.zeros(GRID, np.float32)
    mask = np.zeros(GRID, np.float32)
    observed = np.zeros(GRID, np.float32)
    for idx in obs:
        pos = int(np.argmin(np.abs(curve["gx"] - curve["x"][idx])))
        mask[pos] = 1
        observed[pos] = (curve["y"][idx] - ymean) / ystd
    try:
        ratio = float(curve["R"])
    except Exception:
        ratio = 0.0
    raw = np.concatenate([
        hvec(curve["family"]), hvec(curve["am"]), hvec(curve["R"]), hvec(curve["test"]),
        np.asarray([0, 0, curve["gx"].mean() / 3, np.ptp(curve["gx"]), len(obs) / 4, ratio], np.float32),
    ])
    stress = np.linspace(-1, 1, GRID, dtype=np.float32)
    static = np.stack([observed, mask, stress])
    control = np.stack([prior, observed, mask, stress])
    return {"static": static, "control": control, "target": target.astype(np.float32), "phys": prior, "mask": mask, "oy": observed, "raw": raw.astype(np.float32), "curve": curve, "obs": np.asarray(obs)}


class DatasetNoPhysics(Dataset):
    def __init__(self, curves, ymean, ystd, repeats, seed, sigma):
        self.curves, self.ymean, self.ystd, self.repeats, self.seed, self.sigma = curves, ymean, ystd, repeats, seed, sigma
    def __len__(self): return len(self.curves) * self.repeats
    def __getitem__(self, index):
        curve = self.curves[index % len(self.curves)]
        rng = np.random.default_rng(self.seed + index * 1000003 + np.random.randint(0, 100000))
        k = int(rng.integers(2, min(5, len(curve["x"]))))
        q = make_case(curve, np.sort(rng.choice(len(curve["x"]), k, replace=False)), self.ymean, self.ystd)
        t = float(rng.uniform(.02, .98)); eps = rng.normal(size=GRID).astype(np.float32)
        st = self.sigma * math.sqrt(t * (1 - t)); z = (1 - t) * q["target"] + st * eps
        values = (np.concatenate([z[None], q["static"]]), q["control"], q["raw"], q["target"], eps, q["phys"], q["mask"], q["oy"])
        return tuple(torch.from_numpy(v.astype(np.float32)) for v in values) + (torch.tensor(t, dtype=torch.float32),)


def train(args):
    device, rank, local, world = setup_device(); torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    train_curves, _, _, split_ids = split_curves(args.data, args.split_seed, manifest=args.manifest)
    yvals = np.concatenate([c["y"] for c in train_curves]); ymean, ystd = float(yvals.mean()), float(yvals.std())
    dataset = DatasetNoPhysics(train_curves, ymean, ystd, args.repeats, args.seed + 17 * rank, args.sigma)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=sampler is None, num_workers=0, drop_last=True)
    base = ScoreBase(args.base).to(device); wrapped = DDP(base, device_ids=[local], broadcast_buffers=False) if world > 1 else base
    base_history = epochs(wrapped, loader, sampler, torch.optim.AdamW(wrapped.parameters(), lr=args.lr, weight_decay=1e-4), args.base_epochs, device, rank, world, "base", args.sigma)
    base = wrapped.module if isinstance(wrapped, DDP) else wrapped
    model = ScoreControl(base).to(device); wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    params = [p for p in wrapped.parameters() if p.requires_grad]
    control_history = epochs(wrapped, loader, sampler, torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4), args.control_epochs, device, rank, world, "control", args.sigma)
    if rank == 0:
        core = wrapped.module if isinstance(wrapped, DDP) else wrapped; args.outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": core.state_dict(), "base": args.base, "ymean": ymean, "ystd": ystd, "sigma": args.sigma, "split_seed": args.split_seed, "split_ids": split_ids}, args.outdir / "model.pt")
        (args.outdir / "train_history.json").write_text(json.dumps({"base": base_history, "control": control_history}))
    if world > 1: dist.barrier()
    cleanup()


@torch.no_grad()
def evaluate(args):
    device, rank, _, world = setup_device()
    if world != 1 or rank != 0: raise RuntimeError("Evaluation uses one device")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, curves, _ = split_curves(args.data, ck["split_seed"], split_ids=ck["split_ids"])
    model = ScoreControl(ScoreBase(ck["base"])).to(device); model.load_state_dict(ck["model"]); model.eval()
    rng = np.random.default_rng(args.seed); output = {"method": "Brownian bridge without physics intermediate", "known_points": {}}
    for k in (2, 3, 4):
        cases = [make_case(c, np.sort(rng.choice(len(c["x"]), k, replace=False)), ck["ymean"], ck["ystd"]) for c in curves if len(c["x"]) >= k + 2]
        truth_all, pred_all, crps_all = [], [], []; energy_all = []; diversity_all = []; coverage = []
        grid_mae = []; slope_error = []; violation = []
        for start in range(0, len(cases), args.batch):
            batch = cases[start:start + args.batch]
            tensors = [torch.tensor(np.stack([q[n] for q in batch]), device=device) for n in ("static", "control", "raw", "phys", "mask", "oy")]
            static, control, raw, phys, mask, oy = tensors
            draws = np.stack([sample(model, static, control, raw, phys, mask, oy, args.steps, ck["sigma"], args.eta).cpu().numpy() for _ in range(args.ensemble)]) * ck["ystd"] + ck["ymean"]
            for j, q in enumerate(batch):
                posterior = draws[:, j]; curve = q["curve"]; hidden = np.setdiff1d(np.arange(len(curve["y"])), q["obs"])
                values = np.stack([np.interp(curve["x"][hidden], curve["gx"], draw) for draw in posterior]); truth = curve["y"][hidden]
                truth_all.extend(truth); pred_all.extend(values.mean(axis=0)); crps_all.extend(empirical_crps(values, truth))
                lo, hi = np.quantile(values, [.05, .95], axis=0); coverage.extend((truth >= lo) & (truth <= hi))
                energy_all.append(curve_energy(posterior, curve["gy"])); diversity_all.append(np.mean(np.abs(posterior[:, None] - posterior[None, :])))
                mean = posterior.mean(axis=0); grid_mae.append(np.mean(np.abs(mean - curve["gy"])))
                slope_error.append(abs(np.polyfit(curve["gx"], mean, 1)[0] - np.polyfit(curve["gx"], curve["gy"], 1)[0])); violation.append(np.mean(np.diff(posterior, axis=1) > 0))
        output["known_points"][str(k)] = {"point": point_metrics(truth_all, pred_all), "crps_logN": float(np.mean(crps_all)), "coverage90": float(np.mean(coverage)), "energy_score": float(np.mean(energy_all)), "diversity": float(np.mean(diversity_all)), "curve": {"grid_mae_logN": float(np.mean(grid_mae)), "slope_error": float(np.mean(slope_error)), "monotonic_violation": float(np.mean(violation))}}
        print(json.dumps({"k": k, **output["known_points"][str(k)]}), flush=True)
    args.output.write_text(json.dumps(output, indent=2)); print(args.output, flush=True); cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--data", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--outdir", type=Path, required=True); p.add_argument("--base-epochs", type=int, default=50); p.add_argument("--control-epochs", type=int, default=70); p.add_argument("--repeats", type=int, default=16); p.add_argument("--batch", type=int, default=56); p.add_argument("--base", type=int, default=48); p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--sigma", type=float, default=.8); p.add_argument("--seed", type=int, default=20261003); p.add_argument("--split-seed", type=int, default=20260928)
    p = sub.add_parser("eval"); p.add_argument("--data", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--steps", type=int, default=9); p.add_argument("--eta", type=float, default=0.0); p.add_argument("--ensemble", type=int, default=16); p.add_argument("--batch", type=int, default=48); p.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args(); train(args) if args.cmd == "train" else evaluate(args)
