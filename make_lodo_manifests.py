#!/usr/bin/env python3
"""Create auditable AM/CMA/Weld leave-one-database-out manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TARGETS = ("am2022_curves", "cma2022_curves", "weld2025_disjoint_curves")


def domain(curve):
    return str(curve.get("database") or curve["id"].split(":", 1)[0]).lower()


def stable_fraction(identifier, seed):
    token = f"{identifier}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") / 2**64


def main(args):
    payload = json.loads(args.data.read_text())
    curves = payload["curves"] if isinstance(payload, dict) else payload
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for target in TARGETS:
        test = [curve["id"] for curve in curves if target in domain(curve)]
        candidates = [curve["id"] for curve in curves if target not in domain(curve)]
        val = [identifier for identifier in candidates if stable_fraction(identifier, args.seed) < args.validation_fraction]
        train = [identifier for identifier in candidates if identifier not in set(val)]
        manifest = {"name": f"LODO-{target}", "excluded": target,
                    "train": train, "val": val, "test": test}
        path = args.output_dir / f"lodo_{target}.json"
        path.write_text(json.dumps(manifest, indent=2))
        trainval_ids = set(train + val); test_ids = set(test)
        trainval_path = args.output_dir / f"lodo_{target}_trainval.json"
        test_path = args.output_dir / f"lodo_{target}_test.json"
        trainval_path.write_text(json.dumps({"source": str(args.data), "curves": [c for c in curves if c["id"] in trainval_ids]}))
        test_path.write_text(json.dumps({"source": str(args.data), "curves": [c for c in curves if c["id"] in test_ids]}))
        summary[target] = {"train": len(train), "val": len(val), "test": len(test),
                           "manifest": str(path), "trainval_data": str(trainval_path),
                           "test_data": str(test_path)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("external_data/merged_five_curves.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("lodo_protocol/manifests"))
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    main(parser.parse_args())
