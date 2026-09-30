#!/usr/bin/env python3
"""Matched ControlNet, early-concatenation, no-physics and no-shape ablations."""
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from control_bridge import BaseUNet, ControlBridge, make_case
from control_score_bridge import split_curves
from deep_bridge import COND_RAW, GRID, Block, cleanup, fit_line, setup_device, time_embed
from extended_metrics import curve_energy, empirical_crps


def remove_physics(case):
    q = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
         for key, value in case.items()}
    q["phys"][:] = 0
    q["control"][0] = 0
    q["raw"][-6:-4] = 0
    return q


class VariantDataset(Dataset):
    def __init__(self, curves, prior, ymean, ystd, repeats, seed, variant):
        self.curves, self.prior, self.ymean, self.ystd = curves, prior, ymean, ystd
        self.repeats, self.seed, self.variant = repeats, seed, variant

    def __len__(self):
        return len(self.curves) * self.repeats

    def __getitem__(self, index):
        curve = self.curves[index % len(self.curves)]
        rng = np.random.default_rng(self.seed + index * 1000003 + np.random.randint(0, 100000))
        known = int(rng.integers(2, min(5, len(curve["x"]) - 1)))
        obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
        case = make_case(curve, obs, self.prior, self.ymean, self.ystd)
        if self.variant == "no_physics":
            case = remove_physics(case)
        t = float(rng.uniform(0.02, 0.98))
        noise = rng.normal(size=GRID).astype(np.float32)
        z = (1 - t) * case["target"] + t * case["phys"] + math.sqrt(t * (1 - t)) * noise
        values = (np.concatenate([z[None], case["static"]]), case["control"], case["raw"], case["target"])
        return tuple(torch.from_numpy(value.astype(np.float32)) for value in values) + (torch.tensor(t),)


class EarlyConcatUNet(nn.Module):
    """Same U-Net widths, with all conditions fused once at the input."""
    def __init__(self, base_model: BaseUNet, cdim=128):
        super().__init__(); base = base_model.base; self.base = base
        self.raw_project = nn.Sequential(nn.Linear(COND_RAW, 64), nn.SiLU(), nn.Linear(64, 4))
        self.tcond = base_model.tcond
        self.stem = nn.Conv1d(12, base, 3, padding=1)
        with torch.no_grad():
            self.stem.weight[:, :4].copy_(base_model.stem.weight)
            self.stem.weight[:, 4:].zero_(); self.stem.bias.copy_(base_model.stem.bias)
        self.b1, self.d1, self.b2, self.d2 = base_model.b1, base_model.d1, base_model.b2, base_model.d2
        self.mid, self.u2, self.u1, self.out = base_model.mid, base_model.u2, base_model.u1, base_model.out

    def forward(self, main, control, t, raw):
        broadcast = self.raw_project(raw)[:, :, None].expand(-1, -1, main.shape[-1])
        x = torch.cat([main, control, broadcast], dim=1)
        cond = self.tcond(time_embed(t)); h1 = self.b1(self.stem(x), cond)
        h2 = self.b2(self.d1(h1), cond); middle = self.mid(self.d2(h2), cond)
        up2 = F.interpolate(middle, size=h2.shape[-1], mode="linear", align_corners=False)
        up2 = self.u2(torch.cat([up2, h2], dim=1), cond)
        up1 = F.interpolate(up2, size=h1.shape[-1], mode="linear", align_corners=False)
        return self.out(self.u1(torch.cat([up1, h1], dim=1), cond)).squeeze(1)


def loss_function(prediction, target, diff_weight):
    loss = F.mse_loss(prediction, target)
    if diff_weight:
        loss = loss + diff_weight * F.l1_loss(prediction[:, 1:] - prediction[:, :-1],
                                               target[:, 1:] - target[:, :-1])
    return loss


def train_epochs(model, loader, sampler, optimizer, epochs, device, rank, world,
                 stage, diff_weight, early_concat=False):
    history, start = [], time.time()
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train(); total = 0.0; count = 0
        for main, control, raw, target, t in loader:
            main, control, raw, target, t = [value.to(device) for value in (main, control, raw, target, t)]
            if stage == "base":
                prediction = model(main, t)
            else:
                prediction = model(main, control, t, raw)
            loss = loss_function(prediction, target, diff_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
            optimizer.step(); total += float(loss.detach()); count += 1
        stat = torch.tensor([total, count], device=device)
        if world > 1:
            dist.all_reduce(stat)
        average = float(stat[0] / stat[1])
        if rank == 0 and (epoch % 10 == 0 or epoch == epochs - 1):
            item = {"stage": stage, "epoch": epoch + 1, "loss": average,
                    "seconds": round(time.time() - start, 1)}
            history.append(item); print(json.dumps(item), flush=True)
    return history


def train(args):
    device, rank, local, world = setup_device()
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    train_curves, validation, test, split_ids = split_curves(
        args.data, args.split_seed, manifest=args.manifest)
    slopes, lives = [], []
    for curve in train_curves:
        _, slope = fit_line(curve["x"], curve["y"], -6, 0)
        slopes.append(slope); lives.extend(curve["y"])
    prior, ymean, ystd = float(np.median(slopes)), float(np.mean(lives)), float(np.std(lives))
    dataset = VariantDataset(train_curves, prior, ymean, ystd, args.repeats,
                             args.seed + 17 * rank, args.variant)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler,
                        shuffle=sampler is None, num_workers=0, drop_last=True)
    diff_weight = 0.0 if args.variant == "no_shape" else args.diff_weight

    base = BaseUNet(args.base).to(device)
    wrapped = DDP(base, device_ids=[local], broadcast_buffers=False) if world > 1 else base
    base_history = train_epochs(wrapped, loader, sampler,
                                torch.optim.AdamW(wrapped.parameters(), lr=args.lr, weight_decay=1e-4),
                                args.base_epochs, device, rank, world, "base", diff_weight)
    base = wrapped.module if isinstance(wrapped, DDP) else wrapped
    if args.variant == "early_concat":
        model = EarlyConcatUNet(base).to(device)
        parameters = model.parameters()
    else:
        model = ControlBridge(base).to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    control_history = train_epochs(wrapped, loader, sampler,
                                   torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4),
                                   args.control_epochs, device, rank, world, "conditional", diff_weight,
                                   args.variant == "early_concat")
    if rank == 0:
        core = wrapped.module if isinstance(wrapped, DDP) else wrapped
        args.outdir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": core.state_dict(), "base": args.base, "prior": prior,
            "ymean": ymean, "ystd": ystd, "seed": args.seed,
            "split_seed": args.split_seed, "split_ids": split_ids,
            "variant": args.variant, "diff_weight": diff_weight,
            "split": {"train": len(train_curves), "validation": len(validation), "test": len(test)},
        }, args.outdir / "model.pt")
        (args.outdir / "train_history.json").write_text(json.dumps({
            "base": base_history, "conditional": control_history}, indent=2))
        print("saved", args.outdir / "model.pt", flush=True)
    if world > 1:
        dist.barrier()
    cleanup()


@torch.no_grad()
def sample(model, variant, static, control, raw, phys, mask, observed, steps, eta):
    z = phys.clone(); batch = len(z)
    for index in range(steps):
        current = 1 - index / steps; following = 1 - (index + 1) / steps
        t = torch.full((batch,), current, device=z.device)
        main = torch.cat([z[:, None], static], dim=1)
        prediction = model(main, control, t, raw)
        z = (1 - following) * prediction + following * phys
        if eta and following > 0:
            z = z + eta * math.sqrt(following * (1 - following) / steps) * torch.randn_like(z)
        z = z * (1 - mask) + observed * mask
    return z


def monotonic_violations(draws):
    return float(np.mean(np.diff(draws, axis=1) > 0))


@torch.no_grad()
def evaluate(args):
    device, rank, _, world = setup_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, _, curves, _ = split_curves(args.data, checkpoint["split_seed"],
                                    split_ids=checkpoint["split_ids"])
    base = BaseUNet(checkpoint["base"])
    if checkpoint["variant"] == "early_concat":
        model = EarlyConcatUNet(base).to(device)
    else:
        model = ControlBridge(base).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    local = curves[rank::world]; records = []
    for curve in local:
        for known in (2, 3, 4):
            if len(curve["x"]) < known + 2:
                continue
            token = f'{curve["id"]}|{known}|{args.mask_seed}'.encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:8], 'little') % (2**32)
            rng = np.random.default_rng(seed)
            obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
            case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
            if checkpoint["variant"] == "no_physics":
                case = remove_physics(case)
            tensors = []
            for name in ("static", "control", "raw", "phys", "mask", "oy"):
                value = torch.tensor(case[name], device=device).unsqueeze(0)
                tensors.append(value.repeat(args.ensemble, *([1] * (value.ndim - 1))))
            start = time.time()
            normalized = sample(model, checkpoint["variant"], *tensors, args.steps, args.eta)
            elapsed = time.time() - start
            draws = normalized.cpu().numpy() * checkpoint["ystd"] + checkpoint["ymean"]
            hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
            values = np.asarray([[np.interp(curve["x"][i], curve["gx"], draw)
                                  for i in hidden] for draw in draws])
            truth = curve["y"][hidden]; prediction = values.mean(axis=0)
            low, high = np.quantile(values, [0.05, 0.95], axis=0)
            mean_grid = draws.mean(axis=0)
            basquin_grid = case["phys"] * checkpoint["ystd"] + checkpoint["ymean"]
            basquin_points = np.asarray([np.interp(curve["x"][i], curve["gx"], basquin_grid)
                                         for i in hidden])
            records.append({
                "curve_id": curve["id"], "known": known,
                "mae_logN": float(np.mean(np.abs(prediction - truth))),
                "grid_mae_logN": float(np.mean(np.abs(mean_grid - curve["gy"]))),
                "slope_error": float(abs(np.polyfit(curve["gx"], mean_grid, 1)[0] -
                                           np.polyfit(curve["gx"], curve["gy"], 1)[0])),
                "monotonic_violation": monotonic_violations(draws),
                "crps_logN": float(np.mean(empirical_crps(values, truth))),
                "energy_score": float(curve_energy(draws, curve["gy"])),
                "coverage90": float(np.mean((truth >= low) & (truth <= high))),
                "seconds_per_curve": elapsed,
                "basquin_mae_logN": float(np.mean(np.abs(basquin_points - truth))),
                "basquin_grid_mae_logN": float(np.mean(np.abs(basquin_grid - curve["gy"]))),
                "basquin_slope_error": float(abs(np.polyfit(curve["gx"], basquin_grid, 1)[0] -
                                                   np.polyfit(curve["gx"], curve["gy"], 1)[0])),
                "basquin_monotonic_violation": float(np.mean(np.diff(basquin_grid) > 0)),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps(records))
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for r in range(world):
            merged.extend(json.loads((args.output_dir / f"rank{r}.json").read_text()))
        summary = {}
        keys = ("mae_logN", "grid_mae_logN", "slope_error", "monotonic_violation",
                "crps_logN", "energy_score", "coverage90", "seconds_per_curve")
        for known in (2, 3, 4):
            rows = [row for row in merged if row["known"] == known]
            summary[str(known)] = {key: float(np.mean([row[key] for row in rows])) for key in keys}
            summary[str(known)]["curves"] = len(rows)
        payload = {"variant": checkpoint["variant"], "seed": checkpoint["seed"],
                   "split_seed": checkpoint["split_seed"], "steps": args.steps,
                   "ensemble": args.ensemble, "eta": args.eta, "summary": summary,
                   "case_records": merged}
        (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({k: v for k, v in payload.items() if k != "case_records"}, indent=2))
    cleanup()


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--outdir", type=Path, required=True)
    train_parser.add_argument("--variant", choices=("full", "early_concat", "no_physics", "no_shape"), required=True)
    train_parser.add_argument("--base-epochs", type=int, default=60)
    train_parser.add_argument("--control-epochs", type=int, default=80)
    train_parser.add_argument("--repeats", type=int, default=16)
    train_parser.add_argument("--batch", type=int, default=48)
    train_parser.add_argument("--base", type=int, default=48)
    train_parser.add_argument("--lr", type=float, default=2e-3)
    train_parser.add_argument("--diff-weight", type=float, default=0.15)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--split-seed", type=int, default=20260928)
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--data", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--steps", type=int, default=9)
    eval_parser.add_argument("--ensemble", type=int, default=32)
    eval_parser.add_argument("--eta", type=float, default=0.5)
    eval_parser.add_argument("--mask-seed", type=int, default=20261012)
    args = parser.parse_args(); train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
