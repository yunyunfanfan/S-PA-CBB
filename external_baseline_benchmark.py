#!/usr/bin/env python3
"""Train non-generative baselines on one source database and test other domains."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from baseline_benchmark import build_training, curve_metrics, feature, models, point_metrics, traditional
from deep_bridge import fit_line, load_curves


def mask_rng(curve_id, known, mask_index, seed):
    token = f"{curve_id}|{known}|{mask_index}|{seed}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.random.default_rng(value)


def main(args):
    train, validation, heldout = load_curves(args.train_data, args.split_seed)
    if args.train_manifest is not None:
        manifest = json.loads(args.train_manifest.read_text())
        allowed = set(manifest["train"])
        train = [curve for curve in train + validation + heldout if curve["id"] in allowed]
    slopes, y_values = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope)
        y_values.extend(curve["y"])
    prior = float(np.median(slopes))
    ymean, ystd = float(np.mean(y_values)), float(np.std(y_values))
    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    fitted, timings = {}, {}
    candidates = models(args.seed)
    if args.skip_mlp:
        candidates.pop("MLP", None)
    for name, model in candidates.items():
        start = time.time()
        model.fit(xtrain, ytrain)
        fitted[name] = model
        timings[name] = time.time() - start
        print(json.dumps({"method": name, "train_seconds": timings[name]}), flush=True)

    output = {
        "training_database": args.train_name,
        "training_curves": len(train),
        "training_cases": len(xtrain),
        "train_seconds": timings,
        "external": {},
    }
    for data_path, data_name in zip(args.test_data, args.test_name):
        split = load_curves(data_path, args.split_seed)
        curves = split[0] + split[1] + split[2]
        domain = {"eligible_curves": len(curves), "known_points": {}}
        for known in (2, 3, 4):
            cases = []
            for curve in curves:
                if len(curve["x"]) < known + 2:
                    continue
                for mask_index in range(args.masks):
                    rng = mask_rng(curve["id"], known, mask_index, args.seed)
                    obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
                    cases.append((curve, obs))
            xeval = np.asarray([feature(curve, obs, ymean, ystd) for curve, obs in cases])
            predictions = {
                name: np.clip(model.predict(xeval) * ystd + ymean, 2, 10)
                for name, model in fitted.items()
            }
            for name in ("Linear", "Polynomial", "PCHIP"):
                predictions[name] = np.asarray([traditional(curve, obs, prior)[name] for curve, obs in cases])
            results = {}
            for name, grids in predictions.items():
                truth_points, predicted_points = [], []
                for (curve, obs), grid in zip(cases, grids):
                    hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
                    truth_points.extend(curve["y"][hidden])
                    predicted_points.extend(np.interp(curve["x"][hidden], curve["gx"], grid))
                results[name] = {
                    "point": point_metrics(truth_points, predicted_points),
                    "curve": curve_metrics([curve for curve, _ in cases], grids),
                }
            domain["known_points"][str(known)] = results
            print(json.dumps({"dataset": data_name, "known": known,
                              "mae": {name: round(value["point"]["mae_logN"], 4) for name, value in results.items()}}), flush=True)
        output["external"][data_name] = domain
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-name", required=True)
    parser.add_argument("--train-manifest", type=Path,
                        help="Optional manifest whose train IDs define the exact fitting set.")
    parser.add_argument("--test-data", type=Path, nargs="+", required=True)
    parser.add_argument("--test-name", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--masks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--split-seed", type=int, default=20260928)
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip the high-cost MLP fit, e.g. for repeated LODO folds.")
    args = parser.parse_args()
    if len(args.test_data) != len(args.test_name):
        parser.error("--test-data and --test-name must have the same length")
    main(args)
