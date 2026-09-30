#!/usr/bin/env python3
"""Screen curve-derived downstream tasks on validation, then audit on test.

Task selection uses validation MAE only.  The untouched test split is reported
after selection; test performance never changes which task is selected.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from downstream_fatigue_strength import invert_log_stress, monotone_decreasing, spearman


def load_banks(pacbb_dir: Path, baseline_path: Path):
    pa = {}
    method_name = None
    for path in sorted(pacbb_dir.glob("pacbb_rank*.pkl.gz")):
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        method_name = payload.get("method", method_name or "PA-CBB")
        for row in payload["rows"]:
            pa[tuple(row["key"])] = row
    with gzip.open(baseline_path, "rb") as handle:
        payload = pickle.load(handle)
    base = {tuple(row["key"]): row["predictions"] for row in payload["rows"]}
    common = sorted(set(pa) & set(base))
    if not common:
        raise RuntimeError("No aligned PA-CBB/baseline cases")
    return [(key, pa[key], base[key]) for key in common], (method_name or "PA-CBB")


def logsum10(values):
    values = np.asarray(values, dtype=np.float64)
    peak = np.max(values, axis=-1, keepdims=True)
    return (peak + np.log10(np.sum(10 ** (values - peak), axis=-1, keepdims=True))).squeeze(-1)


def scalar_tasks(gx, curves):
    """Return engineering functionals for one curve or an ensemble."""
    y = np.asarray(curves, dtype=np.float64)
    one = y.ndim == 1
    if one:
        y = y[None, :]
    n = y.shape[-1]
    idx = {q: int(round(q * (n - 1))) for q in (0.10, 0.25, 0.50, 0.75, 0.90)}
    out = {
        "life_at_low_stress": y[:, idx[0.25]],
        "life_at_mid_stress": y[:, idx[0.50]],
        "life_at_high_stress": y[:, idx[0.75]],
        "mean_log_life": np.trapz(y, x=np.linspace(0, 1, n), axis=-1),
        "low_stress_reserve": y[:, : n // 3].mean(axis=-1),
        "high_stress_reserve": y[:, -n // 3 :].mean(axis=-1),
        "life_dynamic_range": y[:, idx[0.10]] - y[:, idx[0.90]],
    }
    # Mission damage per normalized block under three stress spectra.  The
    # constant block size cancels when comparing log damage and mission life.
    mission_indices = np.asarray([idx[0.25], idx[0.50], idx[0.75]])
    spectra = {
        "mission_balanced_log_damage": np.asarray([0.25, 0.50, 0.25]),
        "mission_high_stress_log_damage": np.asarray([0.10, 0.25, 0.65]),
        "mission_low_stress_log_damage": np.asarray([0.65, 0.25, 0.10]),
    }
    for name, weights in spectra.items():
        out[name] = logsum10(np.log10(weights)[None, :] - y[:, mission_indices])
    # Strength at three common target lives, clipped only when outside the
    # curve support; task eligibility is decided from the true curve below.
    for target in (5.0, 6.0, 7.0):
        out[f"strength_N{int(10**target)}"] = np.asarray([
            invert_log_stress(gx, monotone_decreasing(row), target) for row in y
        ])
    if one:
        return {key: float(value[0]) for key, value in out.items()}
    return out


def eligible(task, truth_curve):
    if task.startswith("strength_N"):
        target = math.log10(int(task.split("N", 1)[1]))
        ref = monotone_decreasing(truth_curve)
        return bool(ref.min() <= target <= ref.max())
    return True


def records_from_banks(rows, method_name="PA-CBB"):
    records = []
    for key, pa, baselines in rows:
        split, curve_id, known = key
        gx, truth_curve = pa["gx"], pa["truth"]
        truth = scalar_tasks(gx, truth_curve)
        pa_values = scalar_tasks(gx, pa["draws"])
        methods = {method_name: {name: float(np.mean(value)) for name, value in pa_values.items()}}
        for method, curve in baselines.items():
            methods[method] = scalar_tasks(gx, curve)
        for task, target in truth.items():
            if not eligible(task, truth_curve):
                continue
            for method, values in methods.items():
                records.append({"split": split, "curve_id": curve_id, "known": known,
                                "task": task, "method": method, "truth": target,
                                "prediction": values[task]})
    return records


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["split"], row["known"], row["task"], row["method"])].append(row)
    summary = []
    for (split, known, task, method), rows in sorted(grouped.items()):
        truth = np.asarray([r["truth"] for r in rows]); pred = np.asarray([r["prediction"] for r in rows])
        summary.append({"split": split, "known": known, "task": task, "method": method,
                        "cases": len(rows), "mae": float(np.mean(np.abs(pred - truth))),
                        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                        "spearman": spearman(truth, pred)})
    return summary


def select(summary, margin, method_name="PA-CBB"):
    methods = sorted({r["method"] for r in summary})
    main_competitors = [m for m in methods if m not in (method_name, "Basquin")]
    selected = []
    validation = [r for r in summary if r["split"] == "validation"]
    test = [r for r in summary if r["split"] == "test"]
    keys = sorted({(r["known"], r["task"]) for r in validation})
    for known, task in keys:
        val = {r["method"]: r for r in validation if r["known"] == known and r["task"] == task}
        tst = {r["method"]: r for r in test if r["known"] == known and r["task"] == task}
        if method_name not in val or any(m not in val for m in main_competitors):
            continue
        best_other = min(main_competitors, key=lambda m: val[m]["mae"])
        gain = (val[best_other]["mae"] - val[method_name]["mae"]) / max(val[best_other]["mae"], 1e-12)
        if gain < margin:
            continue
        test_best = min(main_competitors, key=lambda m: tst[m]["mae"])
        test_gain = (tst[test_best]["mae"] - tst[method_name]["mae"]) / max(tst[test_best]["mae"], 1e-12)
        selected.append({
            "known": known, "task": task, "validation_best_baseline": best_other,
            "validation_pacbb_mae": val[method_name]["mae"],
            "validation_baseline_mae": val[best_other]["mae"],
            "validation_relative_gain": gain,
            "test_best_baseline": test_best,
            "test_pacbb_mae": tst[method_name]["mae"],
            "test_baseline_mae": tst[test_best]["mae"],
            "test_relative_gain": test_gain,
            "test_pacbb_spearman": tst[method_name]["spearman"],
            "test_baseline_spearman": tst[test_best]["spearman"],
            "test_confirmed": test_gain > 0,
            "basquin_test_mae": tst.get("Basquin", {}).get("mae"),
        })
    return sorted(selected, key=lambda row: row["validation_relative_gain"], reverse=True)


def main(args):
    rows, method_name = load_banks(args.pacbb_dir, args.baselines)
    records = records_from_banks(rows, method_name)
    summary = summarize(records)
    selected = select(summary, args.margin, method_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_task_metrics.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "validation_selected_tasks.json").write_text(json.dumps({
        "selection_rule": f"{method_name} validation MAE at least margin better than every non-physics main baseline; test is audit only",
        "method": method_name, "margin": args.margin, "selected": selected,
    }, indent=2))
    if selected:
        with (args.output_dir / "validation_selected_tasks.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader(); writer.writerows(selected)
    print(json.dumps({"aligned_cases": len(rows), "candidate_metrics": len(summary),
                      "selected": len(selected), "test_confirmed": sum(r["test_confirmed"] for r in selected)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacbb-dir", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.02)
    main(parser.parse_args())
