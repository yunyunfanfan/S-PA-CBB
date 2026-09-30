#!/usr/bin/env python3
"""Generate the expanded AM2022 source-paper-holdout Table 2."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark" / "table2_source_benchmark.csv"
TEX = ROOT / "paper_elsevier_draft" / "tables" / "table2_source_benchmark.tex"

METHODS = ["PA-CBB", "Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP",
           "Linear", "Polynomial", "PCHIP", "Direct U-Net"]
SHORT = {"PA-CBB": "PA-CBB (ours)", "RandomForest": "RF", "ExtraTrees": "ET",
         "Polynomial": "Poly.", "Direct U-Net": "Direct U-Net"}
METRICS = [
    ("mae", False), ("rmse", False), ("median_ae", False), ("p90_ae", False),
    ("r2", True), ("pearson", True), ("factor2", True), ("factor3", True),
]


def unpack(point: dict) -> dict[str, float]:
    return {
        "mae": float(point["mae_logN"]),
        "rmse": float(point["rmse_logN"]),
        "median_ae": float(point["median_ae_logN"]),
        "p90_ae": float(point["p90_ae_logN"]),
        "r2": float(point["r2"]),
        "pearson": float(point["pearson_r"]),
        "factor2": float(point["factor_2_accuracy"]),
        "factor3": float(point["factor_3_accuracy"]),
    }


def ranks(values: dict[str, float], higher: bool) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=higher)
    result = {}
    for index, (method, value) in enumerate(ordered):
        if index and np.isclose(value, ordered[index - 1][1], rtol=1e-12, atol=1e-12):
            result[method] = result[ordered[index - 1][0]]
        else:
            result[method] = float(index + 1)
    return result


def build() -> list[dict]:
    proposed = json.loads((ROOT / "strict7/source_paper/extended_metrics.json").read_text())
    baseline = json.loads((ROOT / "benchmark/source_paper_baselines.json").read_text())
    unet = json.loads((ROOT / "benchmark/direct_unet_metrics.json").read_text())
    rows = []
    for known in (2, 3, 4):
        block = {"PA-CBB": unpack(proposed["known_points"][str(known)]["point_model"])}
        for method in METHODS[1:-1]:
            block[method] = unpack(baseline["known_points"][str(known)][method]["point"])
        block["Direct U-Net"] = unpack(unet["known_points"][str(known)]["point"])

        rank_sum = {method: 0.0 for method in METHODS}
        metric_ranks = {}
        for metric, higher in METRICS:
            metric_ranks[metric] = ranks({method: block[method][metric] for method in METHODS}, higher)
            for method in METHODS:
                rank_sum[method] += metric_ranks[metric][method]
        for method in METHODS:
            rows.append({
                "k": known,
                "Method": method,
                **block[method],
                "mean_rank": rank_sum[method] / len(METRICS),
                **{f"rank_{metric}": metric_ranks[metric][method] for metric, _ in METRICS},
            })
    return rows


def decorated(value: float, rank: float) -> str:
    text = f"{value:.3f}"
    if rank == 1:
        return r"\textbf{" + text + "}"
    if rank == 2:
        return r"\underline{" + text + "}"
    return text


def write(rows: list[dict]) -> None:
    fields = ["k", "Method"] + [metric for metric, _ in METRICS] + ["mean_rank"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    lines = [
        r"\begin{table*}[!t]",
        r"\caption{AM2022 source-paper-held-out comparison with identical sparse masks, reported separately for two, three and four observed anchors. All baselines predict directly from sparse observations and metadata without a physical intermediate state. Bold and underline denote first and second place within each panel. Basquin-only and bridge-only variants are reserved for ablation.}",
        r"\label{tab:main_v2}",
        r"\centering\setlength{\tabcolsep}{2.15pt}\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{9}{c}}",
        r"\toprule",
        r"Method & MAE $\downarrow$ & RMSE $\downarrow$ & MedAE $\downarrow$ & P90AE $\downarrow$ & $R^2\uparrow$ & Pearson $r\uparrow$ & F2 $\uparrow$ & F3 $\uparrow$ & Mean rank $\downarrow$ \\",
    ]
    for panel, known in enumerate((2, 3, 4)):
        lines += [
            r"\midrule",
            r"\multicolumn{10}{c}{\textbf{(" + chr(97 + panel) + ") $k=" + str(known) + r"$ observed anchors}} \\",
            r"\midrule",
        ]
        for row in [item for item in rows if item["k"] == known]:
            label = SHORT.get(row["Method"], row["Method"])
            if row["Method"] == "PA-CBB":
                label = r"\textbf{" + label + "}"
            values = [decorated(row[metric], row[f"rank_{metric}"]) for metric, _ in METRICS]
            mean_rank = f"{row['mean_rank']:.2f}"
            best_mean = min(item["mean_rank"] for item in rows if item["k"] == known)
            if np.isclose(row["mean_rank"], best_mean):
                mean_rank = r"\textbf{" + mean_rank + "}"
            lines.append(label + " & " + " & ".join(values + [mean_rank]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    data = build()
    write(data)
    for known in (2, 3, 4):
        ours = next(row for row in data if row["k"] == known and row["Method"] == "PA-CBB")
        wins = sum(ours[f"rank_{metric}"] == 1 for metric, _ in METRICS)
        print(f"k={known}: PA-CBB wins {wins}/8 metrics; mean rank={ours['mean_rank']:.2f}")
