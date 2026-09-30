#!/usr/bin/env python3
"""Aggregate ablation seeds and paired curve-bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("mae_logN", "grid_mae_logN", "slope_error", "monotonic_violation",
           "crps_logN", "energy_score", "coverage90", "seconds_per_curve")


def load(root, variant):
    runs = []
    for path in sorted(root.glob(f"{variant}_seed*/eval/metrics.json")):
        payload = json.loads(path.read_text())
        runs.append((int(payload["seed"]), payload["case_records"]))
    return runs


def seed_summary(runs, variant):
    rows = []
    for known in (2, 3, 4):
        seed_means = {metric: [] for metric in METRICS}
        for seed, records in runs:
            selected = [row for row in records if row["known"] == known]
            for metric in METRICS:
                seed_means[metric].append(float(np.mean([row[metric] for row in selected])))
        item = {"variant": variant, "known": known, "seeds": len(runs)}
        for metric, values in seed_means.items():
            item[metric] = float(np.mean(values))
            item[f"{metric}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(item)
    return rows


def paired_bootstrap(full_runs, other_runs, metric, known, repeats, seed):
    paired_seeds = sorted(set(s for s, _ in full_runs) & set(s for s, _ in other_runs))
    differences = []
    for current_seed in paired_seeds:
        left = next(records for s, records in full_runs if s == current_seed)
        right = next(records for s, records in other_runs if s == current_seed)
        left = {(row["curve_id"], row["known"]): row for row in left}
        right = {(row["curve_id"], row["known"]): row for row in right}
        keys = sorted(key for key in left.keys() & right.keys() if key[1] == known)
        differences.extend([(key[0], current_seed, right[key][metric] - left[key][metric]) for key in keys])
    if not differences:
        return {"mean_delta_other_minus_full": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    by_curve = {}
    for curve_id, current_seed, delta in differences:
        by_curve.setdefault(curve_id, []).append(delta)
    curves = sorted(by_curve); curve_values = np.asarray([np.mean(by_curve[curve]) for curve in curves])
    rng = np.random.default_rng(seed); boot = []
    for _ in range(repeats):
        indices = rng.integers(0, len(curve_values), len(curve_values))
        boot.append(float(curve_values[indices].mean()))
    return {
        "mean_delta_other_minus_full": float(curve_values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "curves": len(curves), "paired_seeds": len(paired_seeds),
    }


def main(args):
    variants = ("full", "no_physics", "early_concat", "no_shape")
    runs = {variant: load(args.root, variant) for variant in variants}
    if not runs["full"]:
        raise RuntimeError("No full-model runs found")
    common_seeds = sorted(set.intersection(*({seed for seed, _ in runs[variant]}
                                              for variant in variants if runs[variant])))
    if args.paired_common_seeds:
        runs = {variant: [(seed, records) for seed, records in values if seed in common_seeds]
                for variant, values in runs.items()}
    summary = [row for variant in variants for row in seed_summary(runs[variant], variant) if runs[variant]]
    comparisons = []
    for variant in variants[1:]:
        if not runs[variant]:
            continue
        for known in (2, 3, 4):
            for metric in ("mae_logN", "grid_mae_logN", "slope_error", "monotonic_violation", "energy_score"):
                interval = paired_bootstrap(runs["full"], runs[variant], metric, known,
                                            args.bootstrap, args.seed + known)
                comparisons.append({"comparison": f"{variant}-minus-full", "known": known,
                                    "metric": metric, **interval})

    # Basquin is deterministic for a fixed mask; use one full run to avoid duplicates.
    _, reference = runs["full"][0]
    basquin = []
    for known in (2, 3, 4):
        selected = [row for row in reference if row["known"] == known]
        basquin.append({
            "variant": "basquin_only", "known": known, "seeds": 1,
            "mae_logN": float(np.mean([row["basquin_mae_logN"] for row in selected])),
            "grid_mae_logN": float(np.mean([row["basquin_grid_mae_logN"] for row in selected])),
            "slope_error": float(np.mean([row["basquin_slope_error"] for row in selected])),
            "monotonic_violation": float(np.mean([row["basquin_monotonic_violation"] for row in selected])),
        })
    summary.extend(basquin)
    payload = {"seed_summary": summary, "paired_curve_bootstrap": comparisons,
               "paired_common_seeds": common_seeds if args.paired_common_seeds else None,
               "bootstrap_replicates": args.bootstrap}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_summary.json").write_text(json.dumps(payload, indent=2))
    with (args.output_dir / "ablation_seed_summary.csv").open("w", newline="") as handle:
        fields = sorted(set().union(*(row.keys() for row in summary)))
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    with (args.output_dir / "ablation_bootstrap_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--paired-common-seeds", action="store_true")
    main(parser.parse_args())
