#!/usr/bin/env python3
"""Traditional and non-physics ML baselines for sparse S-N completion."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from control_score_bridge import split_curves
from deep_bridge import GRID, fit_line, hvec


def observed_xy(curve, obs):
    x = curve["x"][obs]
    y = curve["y"][obs]
    ux = np.unique(x)
    uy = np.asarray([np.median(y[x == value]) for value in ux])
    return ux, uy


def linear_extrapolate(x, y, grid):
    if len(x) == 1:
        return np.full_like(grid, y[0])
    out = np.interp(grid, x, y)
    left = grid < x[0]
    right = grid > x[-1]
    out[left] = y[0] + (grid[left] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    out[right] = y[-1] + (grid[right] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def traditional(curve, obs, prior):
    x, y = observed_xy(curve, obs)
    grid = curve["gx"]
    if len(x) < 2:
        linear = np.full_like(grid, y[0])
        return {"Linear": linear, "Polynomial": linear, "PCHIP": linear, "Basquin": linear}
    linear = linear_extrapolate(x, y, grid)
    degree = min(2, len(x) - 1)
    polynomial = np.polyval(np.polyfit(x, y, degree), grid)
    try:
        pchip = PchipInterpolator(x, y, extrapolate=True)(grid)
    except ValueError:
        pchip = linear
    a, b = fit_line(curve["x"][obs], curve["y"][obs], prior)
    basquin = a + b * grid
    return {
        "Linear": np.clip(linear, 2, 10),
        "Polynomial": np.clip(polynomial, 2, 10),
        "PCHIP": np.clip(pchip, 2, 10),
        "Basquin": np.clip(basquin, 2, 10),
    }


def feature(curve, obs, ymean, ystd):
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
    context = np.concatenate(
        [
            hvec(curve["family"]),
            hvec(curve["am"]),
            hvec(curve["R"]),
            hvec(curve["test"]),
            np.asarray([curve["gx"].mean() / 3, np.ptp(curve["gx"]), len(obs) / 4, ratio], np.float32),
        ]
    )
    return np.concatenate([values, mask, context]).astype(np.float32)


def point_metrics(truth, pred):
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    err = pred - truth
    ae = np.abs(err)
    denom = np.sum((truth - truth.mean()) ** 2)
    return {
        "mae_logN": float(ae.mean()),
        "rmse_logN": float(np.sqrt(np.mean(err**2))),
        "median_ae_logN": float(np.median(ae)),
        "p90_ae_logN": float(np.quantile(ae, 0.9)),
        "mean_bias_logN": float(err.mean()),
        "r2": float(1 - np.sum(err**2) / denom),
        "pearson_r": float(np.corrcoef(truth, pred)[0, 1]),
        "factor_2_accuracy": float(np.mean(ae <= math.log10(2))),
        "factor_3_accuracy": float(np.mean(ae <= math.log10(3))),
    }


def curve_metrics(curves, predictions):
    grid_mae = []
    slope_error = []
    violation = []
    for curve, pred in zip(curves, predictions):
        truth = curve["gy"]
        grid_mae.append(np.mean(np.abs(pred - truth)))
        slope_error.append(abs(np.polyfit(curve["gx"], pred, 1)[0] - np.polyfit(curve["gx"], truth, 1)[0]))
        violation.append(np.mean(np.diff(pred) > 0))
    return {
        "grid_mae_logN": float(np.mean(grid_mae)),
        "slope_error": float(np.mean(slope_error)),
        "monotonic_violation": float(np.mean(violation)),
    }


def build_training(curves, prior, ymean, ystd, repeats, seed):
    rng = np.random.default_rng(seed)
    features, targets = [], []
    for curve in curves:
        for _ in range(repeats):
            k = int(rng.integers(2, 5))
            if len(curve["x"]) < k:
                continue
            obs = np.sort(rng.choice(len(curve["x"]), k, replace=False))
            features.append(feature(curve, obs, ymean, ystd))
            targets.append((curve["gy"] - ymean) / ystd)
    return np.asarray(features), np.asarray(targets)


def models(seed):
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=8.0)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance", p=2, n_jobs=-1)),
        "RandomForest": RandomForestRegressor(n_estimators=240, min_samples_leaf=2, max_features=0.70, n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.75, n_jobs=-1, random_state=seed),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 192), activation="relu", alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=260, early_stopping=True, validation_fraction=0.12, n_iter_no_change=18, random_state=seed, verbose=False),
        ),
    }


def main(args):
    train, _, test, _ = split_curves(args.data, args.split_seed, manifest=args.manifest)
    slopes, yvals = [], []
    for curve in train:
        _, slope = fit_line(curve["x"], curve["y"], -6)
        slopes.append(slope)
        yvals.extend(curve["y"])
    prior = float(np.median(slopes))
    ymean, ystd = float(np.mean(yvals)), float(np.std(yvals))
    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    fitted = {}
    timings = {}
    for name, model in models(args.seed).items():
        start = time.time()
        model.fit(xtrain, ytrain)
        timings[name] = time.time() - start
        fitted[name] = model
        print(json.dumps({"method": name, "train_seconds": timings[name]}), flush=True)

    rng = np.random.default_rng(args.seed)
    output = {"split": "source_paper", "train_curves": len(train), "test_curves": len(test), "training_cases": len(xtrain), "known_points": {}, "train_seconds": timings}
    for k in (2, 3, 4):
        cases = []
        for curve in test:
            if len(curve["x"]) < k + 2:
                continue
            obs = np.sort(rng.choice(len(curve["x"]), k, replace=False))
            cases.append((curve, obs))
        xeval = np.asarray([feature(c, obs, ymean, ystd) for c, obs in cases])
        predictions = {}
        for name, model in fitted.items():
            predictions[name] = np.clip(model.predict(xeval) * ystd + ymean, 2, 10)
        for name in ("Linear", "Polynomial", "PCHIP", "Basquin"):
            predictions[name] = np.asarray([traditional(c, obs, prior)[name] for c, obs in cases])

        result = {}
        for name, grids in predictions.items():
            truth_points, pred_points = [], []
            for (curve, obs), grid_pred in zip(cases, grids):
                hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
                truth_points.extend(curve["y"][hidden])
                pred_points.extend(np.interp(curve["x"][hidden], curve["gx"], grid_pred))
            result[name] = {
                "point": point_metrics(truth_points, pred_points),
                "curve": curve_metrics([c for c, _ in cases], grids),
            }
        output["known_points"][str(k)] = result
        print(json.dumps({"k": k, "mae": {name: round(values["point"]["mae_logN"], 4) for name, values in result.items()}}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--split-seed", type=int, default=20260928)
    main(parser.parse_args())
