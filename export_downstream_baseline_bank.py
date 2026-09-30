#!/usr/bin/env python3
"""Export aligned deterministic curve predictions for downstream screening."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import pickle
from pathlib import Path

import numpy as np

from baseline_benchmark import build_training, feature, models, traditional
from control_score_bridge import split_curves
from deep_bridge import fit_line


def mask_for(curve, known, seed):
    token = f"{curve['id']}|{known}|{seed}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31 - 1)
    return np.sort(np.random.default_rng(value).choice(len(curve["x"]), known, replace=False))


def main(args):
    train, validation, test, _ = split_curves(args.data, args.split_seed)
    slopes, lives = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope)
        lives.extend(curve["y"])
    prior = float(np.median(slopes))
    ymean, ystd = float(np.mean(lives)), float(np.std(lives))
    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    all_models = models(args.seed)
    selected = {name: all_models[name] for name in ("ExtraTrees",)}
    for model in selected.values():
        model.fit(xtrain, ytrain)

    rows = []
    for split_name, curves in (("validation", validation), ("test", test)):
        for known in (2, 3, 4):
            cases = [(curve, mask_for(curve, known, args.mask_seed))
                     for curve in curves if len(curve["x"]) >= known + 2]
            xeval = np.asarray([feature(curve, obs, ymean, ystd) for curve, obs in cases])
            learned = {name: np.clip(model.predict(xeval) * ystd + ymean, 2, 10)
                       for name, model in selected.items()}
            for index, (curve, obs) in enumerate(cases):
                trad = traditional(curve, obs, prior)
                predictions = {name: values[index] for name, values in learned.items()}
                predictions.update({name: trad[name] for name in ("PCHIP", "Basquin")})
                rows.append({
                    "key": (split_name, curve["id"], known),
                    "predictions": {name: np.asarray(value, np.float32)
                                    for name, value in predictions.items()},
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wb") as handle:
        pickle.dump({"methods": ["ExtraTrees", "PCHIP", "Basquin"],
                     "split_seed": args.split_seed, "mask_seed": args.mask_seed,
                     "rows": rows}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print({"train": len(train), "validation": len(validation), "test": len(test),
           "rows": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--split-seed", type=int, default=20260928)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    main(parser.parse_args())
