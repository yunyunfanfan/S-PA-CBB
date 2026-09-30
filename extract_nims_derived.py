#!/usr/bin/env python3
"""Extract the openly redistributed NIMS-origin gigacycle subset from Weld2025.

This is not a bulk export of the access-controlled NIMS FDS website.  The eight
records were republished in an open journal article and are present in the
CC-BY Weld2025 database with their DOI and provenance.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("nims_output", type=Path)
    parser.add_argument("weld_output", type=Path)
    args = parser.parse_args()
    obj = json.loads(args.source.read_text())
    nims, weld = [], []
    for curve in obj["curves"]:
        meta = curve.get("metadata", {})
        doi = str(meta.get("doi", "")).lower()
        title = str(meta.get("title", "")).lower()
        if doi == "10.1016/j.stam.2007.09.009" and "gigacycle fatigue data sheets" in title:
            item = dict(curve)
            item["id"] = item["id"].replace("WELD2025:", "NIMS-DERIVED:")
            item["material_family"] = "NIMS-origin"
            nims.append(item)
        else:
            weld.append(curve)
    args.nims_output.write_text(json.dumps({
        "source": "https://doi.org/10.1016/j.stam.2007.09.009",
        "provenance": "Open NIMS-origin gigacycle records redistributed through Weld2025; not a scrape of fds.nims.go.jp",
        "license": "CC BY 4.0 for the Weld2025 redistribution; cite both sources",
        "curves": nims,
    }, ensure_ascii=False))
    args.weld_output.write_text(json.dumps({
        "source": obj.get("source"),
        "license": obj.get("license"),
        "conversion_notes": "NIMS-origin gigacycle records were removed into a disjoint external subset.",
        "curves": weld,
    }, ensure_ascii=False))
    print(json.dumps({"nims_curves": len(nims), "nims_points": sum(len(c["points"]) for c in nims),
                      "remaining_weld_curves": len(weld)}, indent=2))


if __name__ == "__main__":
    main()
