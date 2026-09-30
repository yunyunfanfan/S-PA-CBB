#!/usr/bin/env python3
"""Direct sparse-to-curve U-Net baseline without a physics intermediate state."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from control_score_bridge import split_curves
from deep_bridge import GRID, HASH_DIM, Block, fit_line, hvec, setup_device, cleanup


RAW_DIM = HASH_DIM * 4 + 4


def make_direct_case(curve, obs, ymean, ystd):
    values = np.zeros(GRID, np.float32)
    mask = np.zeros(GRID, np.float32)
    for idx in obs:
        pos = int(np.argmin(np.abs(curve["gx"] - curve["x"][idx])))
        values[pos] = (curve["y"][idx] - ymean) / ystd
        mask[pos] = 1
    try:
        ratio = float(curve["R"])
    except Exception:
        ratio = 0.0
    raw = np.concatenate([
        hvec(curve["family"]), hvec(curve["am"]), hvec(curve["R"]), hvec(curve["test"]),
        np.asarray([curve["gx"].mean() / 3, np.ptp(curve["gx"]), len(obs) / 4, ratio], np.float32),
    ])
    channels = np.stack([values, mask, np.linspace(-1, 1, GRID, dtype=np.float32)])
    target = (curve["gy"] - ymean) / ystd
    return {"channels": channels, "target": target.astype(np.float32), "mask": mask, "values": values, "raw": raw.astype(np.float32), "curve": curve, "obs": np.asarray(obs)}


class DirectDataset(Dataset):
    def __init__(self, curves, ymean, ystd, repeats, seed):
        self.curves, self.ymean, self.ystd, self.repeats, self.seed = curves, ymean, ystd, repeats, seed
    def __len__(self):
        return len(self.curves) * self.repeats
    def __getitem__(self, index):
        curve = self.curves[index % len(self.curves)]
        rng = np.random.default_rng(self.seed + index * 1000003 + np.random.randint(0, 100000))
        k = int(rng.integers(2, min(5, len(curve["x"]))))
        case = make_direct_case(curve, np.sort(rng.choice(len(curve["x"]), k, replace=False)), self.ymean, self.ystd)
        return tuple(torch.from_numpy(case[name]) for name in ("channels", "raw", "target", "mask", "values"))


class DirectUNet(nn.Module):
    def __init__(self, base=48, cdim=128):
        super().__init__()
        self.base = base
        self.cond = nn.Sequential(nn.Linear(RAW_DIM, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.stem = nn.Conv1d(3, base, 3, padding=1)
        self.b1 = Block(base, base, cdim); self.d1 = nn.Conv1d(base, base * 2, 4, 2, 1)
        self.b2 = Block(base * 2, base * 2, cdim); self.d2 = nn.Conv1d(base * 2, base * 4, 4, 2, 1)
        self.mid = Block(base * 4, base * 4, cdim)
        self.u2 = Block(base * 6, base * 2, cdim); self.u1 = Block(base * 3, base, cdim)
        self.out = nn.Sequential(nn.GroupNorm(4, base), nn.SiLU(), nn.Conv1d(base, 1, 3, padding=1))
    def forward(self, channels, raw):
        cond = self.cond(raw)
        h1 = self.b1(self.stem(channels), cond)
        h2 = self.b2(self.d1(h1), cond)
        mid = self.mid(self.d2(h2), cond)
        up2 = F.interpolate(mid, size=h2.shape[-1], mode="linear", align_corners=False)
        up2 = self.u2(torch.cat([up2, h2], dim=1), cond)
        up1 = F.interpolate(up2, size=h1.shape[-1], mode="linear", align_corners=False)
        return self.out(self.u1(torch.cat([up1, h1], dim=1), cond)).squeeze(1)


def point_metrics(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    err = pred - truth; ae = np.abs(err)
    denom = np.sum((truth - truth.mean()) ** 2)
    return {"mae_logN": float(ae.mean()), "rmse_logN": float(np.sqrt(np.mean(err**2))), "median_ae_logN": float(np.median(ae)), "p90_ae_logN": float(np.quantile(ae, .9)), "mean_bias_logN": float(err.mean()), "r2": float(1 - np.sum(err**2) / denom), "pearson_r": float(np.corrcoef(truth, pred)[0, 1]), "factor_2_accuracy": float(np.mean(ae <= math.log10(2))), "factor_3_accuracy": float(np.mean(ae <= math.log10(3)))}


def train(args):
    device, rank, local, world = setup_device()
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    train_curves, val_curves, test_curves, split_ids = split_curves(args.data, args.split_seed, manifest=args.manifest)
    yvals = np.concatenate([c["y"] for c in train_curves]); ymean, ystd = float(yvals.mean()), float(yvals.std())
    dataset = DirectDataset(train_curves, ymean, ystd, args.repeats, args.seed + 17 * rank)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=sampler is None, num_workers=0, drop_last=True)
    model = DirectUNet(args.base).to(device)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    opt = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []; start = time.time()
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        wrapped.train(); total = 0.0; count = 0
        for channels, raw, target, mask, values in loader:
            channels, raw, target, mask, values = [v.to(device) for v in (channels, raw, target, mask, values)]
            pred = wrapped(channels, raw)
            pred = pred * (1 - mask) + values * mask
            loss = F.mse_loss(pred, target) + 0.12 * F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 2.0); opt.step()
            total += float(loss.detach()); count += 1
        stat = torch.tensor([total, count], device=device)
        if world > 1: dist.all_reduce(stat)
        avg = float(stat[0] / stat[1])
        if rank == 0 and (epoch % 10 == 0 or epoch == args.epochs - 1):
            history.append([epoch + 1, avg]); print(json.dumps({"epoch": epoch + 1, "loss": avg, "seconds": round(time.time() - start, 1)}), flush=True)
    if rank == 0:
        core = wrapped.module if isinstance(wrapped, DDP) else wrapped
        args.outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": core.state_dict(), "base": args.base, "ymean": ymean, "ystd": ystd, "seed": args.seed, "split_seed": args.split_seed, "split_ids": split_ids}, args.outdir / "model.pt")
        (args.outdir / "train_history.json").write_text(json.dumps(history))
    if world > 1: dist.barrier()
    cleanup()


@torch.no_grad()
def evaluate(args):
    device, rank, _, world = setup_device()
    if world != 1 or rank != 0: raise RuntimeError("Evaluation uses one device.")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, curves, _ = split_curves(args.data, ck["split_seed"], split_ids=ck["split_ids"])
    model = DirectUNet(ck["base"]).to(device); model.load_state_dict(ck["model"]); model.eval()
    rng = np.random.default_rng(args.seed); output = {"method": "Direct conditional U-Net (no physics state)", "known_points": {}}
    for k in (2, 3, 4):
        cases = [make_direct_case(c, np.sort(rng.choice(len(c["x"]), k, replace=False)), ck["ymean"], ck["ystd"]) for c in curves if len(c["x"]) >= k + 2]
        grids = []
        for start in range(0, len(cases), args.batch):
            batch = cases[start:start + args.batch]
            channels = torch.tensor(np.stack([q["channels"] for q in batch]), device=device)
            raw = torch.tensor(np.stack([q["raw"] for q in batch]), device=device)
            mask = torch.tensor(np.stack([q["mask"] for q in batch]), device=device)
            values = torch.tensor(np.stack([q["values"] for q in batch]), device=device)
            pred = model(channels, raw) * (1 - mask) + values * mask
            grids.extend(pred.cpu().numpy() * ck["ystd"] + ck["ymean"])
        truths, preds = [], []
        grid_mae, slope_error, violation = [], [], []
        for q, pred in zip(cases, grids):
            curve = q["curve"]; hidden = np.setdiff1d(np.arange(len(curve["y"])), q["obs"])
            truths.extend(curve["y"][hidden]); preds.extend(np.interp(curve["x"][hidden], curve["gx"], pred))
            grid_mae.append(np.mean(np.abs(pred - curve["gy"])))
            slope_error.append(abs(np.polyfit(curve["gx"], pred, 1)[0] - np.polyfit(curve["gx"], curve["gy"], 1)[0]))
            violation.append(np.mean(np.diff(pred) > 0))
        output["known_points"][str(k)] = {"point": point_metrics(truths, preds), "curve": {"grid_mae_logN": float(np.mean(grid_mae)), "slope_error": float(np.mean(slope_error)), "monotonic_violation": float(np.mean(violation))}}
        print(json.dumps({"k": k, **output["known_points"][str(k)]}), flush=True)
    args.output.write_text(json.dumps(output, indent=2)); print(args.output, flush=True); cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--data", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--outdir", type=Path, required=True); p.add_argument("--epochs", type=int, default=90); p.add_argument("--repeats", type=int, default=16); p.add_argument("--batch", type=int, default=64); p.add_argument("--base", type=int, default=48); p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--seed", type=int, default=20261003); p.add_argument("--split-seed", type=int, default=20260928)
    p = sub.add_parser("eval"); p.add_argument("--data", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--batch", type=int, default=64); p.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args(); train(args) if args.cmd == "train" else evaluate(args)
