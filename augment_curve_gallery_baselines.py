#!/usr/bin/env python3
"""Attach matched PCHIP, Ridge and Direct U-Net predictions to gallery cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline_benchmark import build_training, feature, traditional
from control_score_bridge import split_curves
from deep_bridge import fit_line
from direct_unet import DirectUNet, make_direct_case


def serial(values):
    return np.asarray(values, dtype=float).tolist()


def hidden_mae(curve, obs, prediction):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
    estimate = np.interp(curve["x"][hidden], curve["gx"], prediction)
    return float(np.mean(np.abs(estimate - curve["y"][hidden])))


@torch.no_grad()
def main(args):
    payload = json.loads(args.input.read_text())
    train, _, _, _ = split_curves(args.data, args.split_seed, manifest=args.manifest)
    slopes, yvals = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope)
        yvals.extend(curve["y"])
    prior = float(np.median(slopes))
    ymean, ystd = float(np.mean(yvals)), float(np.std(yvals))

    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=8.0))
    ridge.fit(xtrain, ytrain)

    direct_ck = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    direct = DirectUNet(direct_ck["base"])
    direct.load_state_dict(direct_ck["model"])
    direct.eval()

    for record in payload["records"]:
        curve = {
            "id": record["id"],
            "name": record["name"],
            "family": record["family"],
            "am": record["am_type"],
            "R": record["load_ratio"],
            "test": record["test_type"],
            "gx": np.asarray(record["gx_log_stress"], dtype=np.float32),
            "gy": np.asarray(record["truth_grid_logN"], dtype=np.float32),
            "x": np.asarray(record["raw_log_stress"], dtype=np.float32),
            "y": np.asarray(record["raw_logN"], dtype=np.float32),
        }
        obs = np.asarray(record["observed_indices"], dtype=int)
        pchip = traditional(curve, obs, prior)["PCHIP"]
        ridge_grid = np.clip(
            ridge.predict(feature(curve, obs, ymean, ystd)[None])[0] * ystd + ymean,
            2,
            10,
        )
        case = make_direct_case(curve, obs, direct_ck["ymean"], direct_ck["ystd"])
        channels = torch.from_numpy(case["channels"])[None]
        raw = torch.from_numpy(case["raw"])[None]
        mask = torch.from_numpy(case["mask"])[None]
        values = torch.from_numpy(case["values"])[None]
        direct_grid = direct(channels, raw) * (1 - mask) + values * mask
        direct_grid = np.clip(
            direct_grid[0].numpy() * direct_ck["ystd"] + direct_ck["ymean"],
            2,
            10,
        )
        predictions = {"PCHIP": pchip, "Ridge": ridge_grid, "Direct U-Net": direct_grid}
        record["comparison_grid_logN"] = {name: serial(values) for name, values in predictions.items()}
        record["comparison_hidden_mae_logN"] = {
            name: hidden_mae(curve, obs, values) for name, values in predictions.items()
        }

    payload["comparison_methods"] = ["PCHIP", "Ridge", "Direct U-Net"]
    payload["comparison_note"] = "All methods use the exact same curve and observed indices as the proposed model."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
