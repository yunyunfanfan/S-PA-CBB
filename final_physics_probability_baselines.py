#!/usr/bin/env python3
"""Aligned weak-physics and probabilistic baselines for sparse S--N completion.

All methods use the same curve split and hashed sparse masks as the final
endpoint-sampler evaluation.  Probabilistic methods are calibrated on the
validation curves and scored once on the untouched test curves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from baseline_benchmark import build_training, feature, point_metrics
from control_score_bridge import split_curves
from deep_bridge import fit_line
from extended_metrics import curve_energy, empirical_crps, interval_score


def mask_for(curve, known, seed):
    token = f'{curve["id"]}|{known}|{seed}'.encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
    return np.sort(np.random.default_rng(value).choice(len(curve["x"]), known, replace=False))


def observed(curve, obs):
    x, y = curve["x"][obs], curve["y"][obs]
    ux = np.unique(x)
    return ux, np.asarray([np.median(y[x == value]) for value in ux])


def clip_curve(values):
    return np.clip(np.asarray(values, np.float64), 2.0, 10.0)


def weak_physics(curve, obs, prior_slope):
    x, y = observed(curve, obs); grid = curve["gx"]
    if len(x) < 2:
        constant = np.full_like(grid, y[0])
        return {name: constant for name in
                ("Naive Basquin", "Extreme-two Basquin", "Fixed-slope Wohler", "Fixed-limit Stromeyer")}

    raw_b, raw_a = np.polyfit(x, y, 1)
    raw = raw_a + raw_b * grid
    edge_b = (y[-1] - y[0]) / max(x[-1] - x[0], 1e-8)
    edge = y[0] + edge_b * (grid - x[0])
    fixed_a = float(np.median(y - prior_slope * x))
    fixed = fixed_a + prior_slope * grid

    # Stromeyer S = S_e + A N^b, inverted after fixing an intentionally simple
    # fatigue-limit estimate from the sparse anchors.
    stress, grid_stress = 10**x, 10**grid
    fatigue_limit = 0.75 * float(np.min(stress))
    transformed = np.log10(np.maximum(stress - fatigue_limit, 1e-6))
    st_b, st_a = np.polyfit(transformed, y, 1)
    stromeyer = st_a + st_b * np.log10(np.maximum(grid_stress - fatigue_limit, 1e-6))
    return {
        "Naive Basquin": clip_curve(raw),
        "Extreme-two Basquin": clip_curve(edge),
        "Fixed-slope Wohler": clip_curve(fixed),
        "Fixed-limit Stromeyer": clip_curve(stromeyer),
    }


def gp_draws(curve, obs, size, seed):
    x, y = observed(curve, obs); grid = curve["gx"]
    centre, scale = float(np.mean(x)), max(float(np.ptp(curve["gx"])), 1e-3)
    xs = ((x - centre) / scale)[:, None]
    gs = ((grid - centre) / scale)[:, None]
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(0.25, length_scale_bounds="fixed") + WhiteKernel(0.03, noise_level_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer=None,
                                     normalize_y=True, random_state=seed)
    model.fit(xs, y)
    return clip_curve(model.sample_y(gs, n_samples=size, random_state=seed).T)


def bayesian_basquin_draws(curve, obs, size, seed, slope_mean, slope_sd, noise_sd, intercept_mean):
    x, y = observed(curve, obs)
    X = np.column_stack([np.ones(len(x)), x])
    prior_mean = np.asarray([intercept_mean, slope_mean])
    prior_var = np.asarray([25.0, max(slope_sd**2, 0.05)])
    precision = np.diag(1.0 / prior_var) + X.T @ X / noise_sd**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (prior_mean / prior_var + X.T @ y / noise_sd**2)
    beta = np.random.default_rng(seed).multivariate_normal(mean, covariance, size=size)
    return clip_curve(beta[:, :1] + beta[:, 1:] * curve["gx"][None])


def hidden_values(curve, obs, draws):
    hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
    values = np.stack([np.interp(curve["x"][hidden], curve["gx"], draw) for draw in draws])
    return hidden, values, curve["y"][hidden]


def probabilistic_records(curves, known, method, draw_fn, mask_seed):
    records = []
    for curve in curves:
        if len(curve["x"]) < known + 2:
            continue
        obs = mask_for(curve, known, mask_seed)
        draws = draw_fn(curve, obs)
        hidden, values, truth = hidden_values(curve, obs, draws)
        lo, hi = np.quantile(values, [0.05, 0.95], axis=0)
        mean_grid = draws.mean(axis=0)
        records.append({
            "curve_id": curve["id"], "method": method, "known": known,
            "truth": truth.tolist(), "prediction": values.mean(0).tolist(),
            "lo": lo.tolist(), "hi": hi.tolist(),
            "grid_mae": float(np.mean(np.abs(mean_grid - curve["gy"]))),
            "slope_error": float(abs(np.polyfit(curve["gx"], mean_grid, 1)[0] - np.polyfit(curve["gx"], curve["gy"], 1)[0])),
            "violation": float(np.mean(np.diff(mean_grid) > 0)),
            "crps": float(np.mean(empirical_crps(values, truth))),
            "energy": float(curve_energy(draws, curve["gy"])),
        })
    return records


def summarize_probability(validation, test):
    val_truth = np.concatenate([np.asarray(row["truth"]) for row in validation])
    val_lo = np.concatenate([np.asarray(row["lo"]) for row in validation])
    val_hi = np.concatenate([np.asarray(row["hi"]) for row in validation])
    scores = np.maximum(val_lo - val_truth, val_truth - val_hi)
    q_level = min(1.0, math.ceil((len(scores) + 1) * 0.90) / len(scores))
    q = float(np.quantile(scores, q_level, method="higher"))
    truth = np.concatenate([np.asarray(row["truth"]) for row in test])
    pred = np.concatenate([np.asarray(row["prediction"]) for row in test])
    lo = np.concatenate([np.asarray(row["lo"]) for row in test])
    hi = np.concatenate([np.asarray(row["hi"]) for row in test])
    raw_score = interval_score(truth, lo, hi, 0.1)
    return {
        "point": point_metrics(truth, pred),
        "grid_mae_logN": float(np.mean([row["grid_mae"] for row in test])),
        "slope_error": float(np.mean([row["slope_error"] for row in test])),
        "monotonic_violation": float(np.mean([row["violation"] for row in test])),
        "crps_logN": float(np.mean([row["crps"] for row in test])),
        "energy_score": float(np.mean([row["energy"] for row in test])),
        "raw_coverage90": float(np.mean((truth >= lo) & (truth <= hi))),
        "raw_width90_logN": float(np.mean(hi - lo)),
        "raw_interval_score90": float(np.mean(raw_score)),
        "conformal_q": q,
        "conformal_coverage90": float(np.mean((truth >= lo - q) & (truth <= hi + q))),
        "conformal_width90_logN": float(np.mean(hi - lo + 2 * q)),
        "curves": len(test),
    }


def summarize_deterministic(curves, known, prior, mask_seed):
    bags = {}
    for curve in curves:
        if len(curve["x"]) < known + 2:
            continue
        obs = mask_for(curve, known, mask_seed)
        hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
        for name, grid in weak_physics(curve, obs, prior).items():
            row = bags.setdefault(name, {"truth": [], "pred": [], "grid": [], "slope": [], "viol": []})
            row["truth"].extend(curve["y"][hidden]); row["pred"].extend(np.interp(curve["x"][hidden], curve["gx"], grid))
            row["grid"].append(np.mean(np.abs(grid - curve["gy"])))
            row["slope"].append(abs(np.polyfit(curve["gx"], grid, 1)[0] - np.polyfit(curve["gx"], curve["gy"], 1)[0]))
            row["viol"].append(np.mean(np.diff(grid) > 0))
    return {name: {"point": point_metrics(row["truth"], row["pred"]),
                   "grid_mae_logN": float(np.mean(row["grid"])),
                   "slope_error": float(np.mean(row["slope"])),
                   "monotonic_violation": float(np.mean(row["viol"]))}
            for name, row in bags.items()}


def main(args):
    start = time.time()
    train, validation, test, _ = split_curves(args.data, args.split_seed, manifest=args.manifest)
    slopes, intercepts, residuals, lives = [], [], [], []
    for curve in train:
        a, b = fit_line(curve["x"], curve["y"], -6, 0)
        slopes.append(b); intercepts.append(a); lives.extend(curve["y"])
        residuals.extend(curve["y"] - (a + b * curve["x"]))
    prior = float(np.median(slopes)); slope_sd = max(float(np.std(slopes)), 0.25)
    noise_sd = max(float(np.median(np.abs(residuals))) * 1.4826, 0.08)
    intercept_mean = float(np.median(intercepts)); ymean, ystd = float(np.mean(lives)), float(np.std(lives))

    xtrain, ytrain = build_training(train, prior, ymean, ystd, args.repeats, args.seed + 11)
    forest = ExtraTreesRegressor(n_estimators=args.ensemble, min_samples_leaf=2, max_features=0.75,
                                 n_jobs=-1, random_state=args.seed).fit(xtrain, ytrain)

    output = {"protocol": "aligned hashed masks; validation-only conformal calibration",
              "train_curves": len(train), "validation_curves": len(validation), "test_curves": len(test),
              "ensemble": args.ensemble, "known_points": {}}
    for known in (2, 3, 4):
        physical = summarize_deterministic(test, known, prior, args.mask_seed)

        def gp(curve, obs):
            return gp_draws(curve, obs, args.ensemble, int.from_bytes(hashlib.sha256(f'gp|{curve["id"]}|{known}|{args.seed}'.encode()).digest()[:4], "little"))

        def bayes(curve, obs):
            seed = int.from_bytes(hashlib.sha256(f'bayes|{curve["id"]}|{known}|{args.seed}'.encode()).digest()[:4], "little")
            return bayesian_basquin_draws(curve, obs, args.ensemble, seed, prior, slope_sd, noise_sd, intercept_mean)

        def trees(curve, obs):
            x = feature(curve, obs, ymean, ystd)[None]
            return clip_curve(np.stack([tree.predict(x)[0] for tree in forest.estimators_]) * ystd + ymean)

        probability = {}
        for name, fn in (("GP-RBF", gp), ("Bayesian Basquin", bayes), ("ExtraTrees ensemble", trees)):
            val_rows = probabilistic_records(validation, known, name, fn, args.mask_seed)
            test_rows = probabilistic_records(test, known, name, fn, args.mask_seed)
            probability[name] = summarize_probability(val_rows, test_rows)
        output["known_points"][str(known)] = {"weak_physics": physical, "probabilistic": probability}
        print(json.dumps({"known": known,
                          "physical_mae": {k: round(v["point"]["mae_logN"], 4) for k, v in physical.items()},
                          "probabilistic_crps": {k: round(v["crps_logN"], 4) for k, v in probability.items()}}), flush=True)
    output["seconds"] = time.time() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--split-seed", type=int, default=20260928)
    parser.add_argument("--mask-seed", type=int, default=20261012)
    main(parser.parse_args())
