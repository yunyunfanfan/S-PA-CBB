#!/usr/bin/env python3
"""Deterministic fatigue-strength inversion baselines on the checkpoint split."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from baseline_benchmark import build_training, feature, models, traditional
from control_score_bridge import split_curves
from deep_bridge import fit_line
from downstream_fatigue_strength import (TARGETS, invert_log_stress,
                                         monotone_decreasing, pairwise_accuracy,
                                         spearman, stable_rng)


def main(args):
    train, _, test, _ = split_curves(args.data, args.split_seed)
    slopes, lives = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope); lives.extend(curve["y"])
    prior, ymean, ystd = float(np.median(slopes)), float(np.mean(lives)), float(np.std(lives))
    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    fitted = {}
    for name, model in models(args.seed).items():
        start = time.time(); model.fit(xtrain, ytrain); fitted[name] = model
        print(json.dumps({"method": name, "seconds": time.time() - start}), flush=True)

    case_records = []
    for known in (2, 3, 4):
        cases = []
        for curve in test:
            if len(curve["x"]) < known + 2:
                continue
            rng = stable_rng(curve["id"], known, 0, args.mask_seed)
            obs = np.sort(rng.choice(len(curve["x"]), known, replace=False))
            cases.append((curve, obs))
        xeval = np.asarray([feature(curve, obs, ymean, ystd) for curve, obs in cases])
        predictions = {name: np.clip(model.predict(xeval) * ystd + ymean, 2, 10)
                       for name, model in fitted.items()}
        for name in ("Linear", "Polynomial", "PCHIP", "Basquin"):
            predictions[name] = np.asarray([traditional(curve, obs, prior)[name]
                                            for curve, obs in cases])
        for method, grids in predictions.items():
            for (curve, obs), grid in zip(cases, grids):
                reference = monotone_decreasing(curve["gy"])
                for target in TARGETS:
                    if not (reference.min() <= target <= reference.max()):
                        continue
                    truth = 10 ** invert_log_stress(curve["gx"], reference, target)
                    prediction = 10 ** invert_log_stress(curve["gx"], grid, target)
                    case_records.append({"method": method, "known": known,
                                         "target_cycles": int(10 ** target),
                                         "curve_id": curve["id"], "truth_mpa": truth,
                                         "prediction_mpa": prediction})
    summary = []
    for method in sorted({row["method"] for row in case_records}):
        for known in (2, 3, 4):
            for target in (100000, 1000000, 10000000):
                rows = [row for row in case_records if row["method"] == method and
                        row["known"] == known and row["target_cycles"] == target]
                if not rows:
                    continue
                truth = np.asarray([row["truth_mpa"] for row in rows]); pred = np.asarray([row["prediction_mpa"] for row in rows])
                summary.append({"method": method, "known": known, "target_cycles": target,
                                "cases": len(rows), "stress_mae_mpa": float(np.mean(np.abs(pred - truth))),
                                "relative_stress_error": float(np.mean(np.abs(pred - truth) / np.maximum(truth, 1e-9))),
                                "spearman_rho": spearman(truth, pred),
                                "pairwise_ranking_accuracy": pairwise_accuracy(truth, pred)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fatigue_strength_baselines.json").write_text(json.dumps({
        "split_seed": args.split_seed, "train_curves": len(train), "test_curves": len(test),
        "training_cases": len(xtrain), "summary": summary}, indent=2))
    with (args.output_dir / "fatigue_strength_baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(args.output_dir / "fatigue_strength_baselines.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--mask-seed", type=int, default=20261010)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
