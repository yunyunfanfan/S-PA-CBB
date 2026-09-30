#!/usr/bin/env python3
"""Recreate the deterministic pooled-model test split and export it by database."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from deep_bridge import load_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _, _, test = load_curves(args.data, args.seed)
    test_ids = {curve["id"] for curve in test}
    raw = json.loads(args.data.read_text())
    selected = [curve for curve in raw["curves"] if curve["id"] in test_ids]
    if len(selected) != len(test_ids):
        counts = Counter(curve["id"] for curve in raw["curves"])
        missing = sorted(test_ids - set(counts))
        raise RuntimeError(f"Could not recover {len(missing)} test curves: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for curve in selected:
        groups.setdefault(curve.get("database", "unknown"), []).append(curve)

    manifest = {
        "source": str(args.data),
        "seed": args.seed,
        "total_test_curves": len(selected),
        "domains": {},
    }
    overall = {"source": str(args.data), "split": "deterministic_test", "curves": selected}
    (args.output_dir / "pooled_test_all.json").write_text(json.dumps(overall, ensure_ascii=False))
    for database, curves in sorted(groups.items()):
        safe = database.lower().replace("-", "_")
        path = args.output_dir / f"pooled_test_{safe}.json"
        path.write_text(json.dumps({
            "source": str(args.data),
            "database": database,
            "split": "deterministic_test",
            "curves": curves,
        }, ensure_ascii=False))
        manifest["domains"][database] = {"curves": len(curves), "file": str(path)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
