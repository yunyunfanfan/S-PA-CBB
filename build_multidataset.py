#!/usr/bin/env python3
"""Audit, deduplicate, and merge curve-level fatigue JSON databases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def signature(curve):
    points = sorted(
        (round(float(p[0]), 5), round(float(p[1]), 2), int(bool(p[2])))
        for p in curve["points"] if float(p[0]) > 0 and float(p[1]) > 0
    )
    return hashlib.sha256(json.dumps(points).encode()).hexdigest()


def eligible(curve):
    points = np.asarray([[p[0], p[1]] for p in curve["points"] if not p[2] and p[0] > 0 and p[1] > 0], float)
    if len(points) < 6 or len(np.unique(points[:, 0])) < 4:
        return False
    x, y = np.log10(points[:, 0]), np.log10(points[:, 1])
    if np.ptp(x) < 0.04:
        return False
    slope = np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-10)
    return -25 < slope < -0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    seen = set()
    merged = []
    report = {"datasets": {}, "duplicates_removed": 0}
    for source_path in args.inputs:
        obj = json.loads(source_path.read_text())
        source_name = source_path.stem
        raw = obj["curves"]
        selected = []
        duplicates = 0
        for curve in raw:
            if not eligible(curve):
                continue
            sig = signature(curve)
            if sig in seen:
                duplicates += 1
                continue
            seen.add(sig)
            curve = dict(curve)
            curve["database"] = source_name
            selected.append(curve)
        merged.extend(selected)
        report["datasets"][source_name] = {
            "raw_curves": len(raw),
            "raw_points": sum(len(curve["points"]) for curve in raw),
            "eligible_curves": len(selected),
            "eligible_points": sum(len(curve["points"]) for curve in selected),
            "duplicates_removed": duplicates,
        }
        report["duplicates_removed"] += duplicates
    report["merged_eligible_curves"] = len(merged)
    report["merged_points"] = sum(len(curve["points"]) for curve in merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source": [str(path) for path in args.inputs],
        "license": "mixed; see source records",
        "curves": merged,
    }, ensure_ascii=False))
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
