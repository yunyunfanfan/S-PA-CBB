#!/usr/bin/env python3
"""Deterministic main baselines on the final hashed sparse masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from baseline_benchmark import (build_training, curve_metrics, feature, models,
                                point_metrics, traditional)
from control_score_bridge import split_curves
from deep_bridge import fit_line


def mask_for(curve, known, seed):
    token = f'{curve["id"]}|{known}|{seed}'.encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.sort(np.random.default_rng(value).choice(len(curve["x"]), known, replace=False))


def main(args):
    train, _, test, _ = split_curves(args.data, args.split_seed, manifest=args.manifest)
    slopes, yvals = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope); yvals.extend(curve["y"])
    prior = float(np.median(slopes)); ymean, ystd = float(np.mean(yvals)), float(np.std(yvals))
    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    fitted, timings = {}, {}
    for name, model in models(args.seed).items():
        tic = time.time(); model.fit(xtrain, ytrain); timings[name] = time.time() - tic; fitted[name] = model
        print(json.dumps({"method": name, "seconds": timings[name]}), flush=True)
    output = {"protocol": "final hashed masks", "mask_seed": args.mask_seed,
              "train_curves": len(train), "test_curves": len(test), "train_seconds": timings,
              "known_points": {}}
    for known in (2, 3, 4):
        cases = []
        for curve in test:
            if len(curve["x"]) >= known + 2:
                cases.append((curve, mask_for(curve, known, args.mask_seed)))
        xeval = np.asarray([feature(curve, obs, ymean, ystd) for curve, obs in cases])
        predictions = {name: np.clip(model.predict(xeval) * ystd + ymean, 2, 10)
                       for name, model in fitted.items()}
        for name in ("Linear", "Polynomial", "PCHIP"):
            predictions[name] = np.asarray([traditional(curve, obs, prior)[name] for curve, obs in cases])
        result = {}
        for name, grids in predictions.items():
            truth, pred = [], []
            for (curve, obs), grid in zip(cases, grids):
                hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
                truth.extend(curve["y"][hidden]); pred.extend(np.interp(curve["x"][hidden], curve["gx"], grid))
            result[name] = {"point": point_metrics(truth, pred),
                            "curve": curve_metrics([curve for curve, _ in cases], grids)}
        output["known_points"][str(known)] = result
        print(json.dumps({"known": known, "mae": {name: round(row["point"]["mae_logN"], 4) for name, row in result.items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261030); parser.add_argument("--split-seed", type=int, default=20260928)
    parser.add_argument("--mask-seed", type=int, default=20261012); main(parser.parse_args())
