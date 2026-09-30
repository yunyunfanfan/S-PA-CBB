#!/usr/bin/env python3
"""Build auditable multi-metric tables for the pooled five-domain test.

All entries are macro-averaged over the same 15 dataset--sparsity tasks.
The detailed CSV preserves every task/method value used by the paper table.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("MULTIDATA_RESULTS", ROOT / "multidataset_results"))
TABLES = RESULTS / "tables"
PAPER_TABLES = ROOT / "paper_elsevier_draft" / "tables"

DOMAINS = ["AM2022", "CMA2022", "HEA2022", "NIMS-derived", "Weld2025"]
SLUGS = {domain: domain.lower().replace("-", "_") for domain in DOMAINS}
METHODS = ["PA-CBB", "Ridge", "kNN", "RandomForest", "ExtraTrees", "MLP", "Linear", "Polynomial", "PCHIP"]
SHORT_METHODS = {
    "PA-CBB": "PA-CBB", "Ridge": "Ridge", "kNN": "kNN", "RandomForest": "RF",
    "ExtraTrees": "ET", "MLP": "MLP", "Linear": "Linear", "Polynomial": "Poly.", "PCHIP": "PCHIP",
}

# key, label, direction (True means larger is better), display precision
METRICS = [
    ("mae", r"MAE $\downarrow$", False, 3),
    ("rmse", r"RMSE $\downarrow$", False, 3),
    ("median_ae", r"MedAE $\downarrow$", False, 3),
    ("p90_ae", r"P90AE $\downarrow$", False, 3),
    ("r2", r"$R^2\uparrow$", True, 3),
    ("pearson", r"Pearson $r\uparrow$", True, 3),
    ("factor2", r"F2 $\uparrow$", True, 3),
    ("factor3", r"F3 $\uparrow$", True, 3),
]


def rank_values(values: dict[str, float], higher: bool) -> dict[str, float]:
    """Competition ranks with exact-value ties."""
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=higher)
    ranks: dict[str, float] = {}
    for idx, (name, value) in enumerate(ordered):
        if idx and np.isclose(value, ordered[idx - 1][1], rtol=1e-12, atol=1e-12):
            ranks[name] = ranks[ordered[idx - 1][0]]
        else:
            ranks[name] = float(idx + 1)
    return ranks


def load_predictions(domain: str) -> dict[str, np.ndarray]:
    arrays = [np.load(path) for path in sorted(
        (RESULTS / "pooled_test_model" / "predictions").glob(f"{SLUGS[domain]}_rank*.npz")
    )]
    if not arrays:
        raise FileNotFoundError(f"No prediction shards found for {domain}")
    return {key: np.concatenate([array[key] for array in arrays]) for key in ("truth", "prediction", "known")}


def proposed_task_metrics(domain: str, known: int, full: dict) -> dict[str, float]:
    pred = load_predictions(domain)
    keep = pred["known"] == known
    truth = pred["truth"][keep].astype(float)
    estimate = pred["prediction"][keep].astype(float)
    errors = estimate - truth
    abs_errors = np.abs(errors)
    record = full[domain]["known_points"][str(known)]
    pearson = float(np.corrcoef(truth, estimate)[0, 1]) if len(truth) > 1 else float("nan")
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_ae": float(np.median(abs_errors)),
        "p90_ae": float(np.quantile(abs_errors, 0.90)),
        "r2": float(record["r2"]),
        "pearson": pearson,
        "factor2": float(record["factor_2_accuracy"]),
        "factor3": float(record["factor_3_accuracy"]),
    }


def baseline_task_metrics(record: dict) -> dict[str, float]:
    point, curve = record["point"], record["curve"]
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


def build_records() -> list[dict]:
    baseline = json.loads((RESULTS / "pooled_test_baselines.json").read_text())
    full = {
        domain: json.loads((RESULTS / "pooled_test_model" / "metrics" / f"{SLUGS[domain]}.json").read_text())
        for domain in DOMAINS
    }
    records = []
    for domain in DOMAINS:
        for known in (2, 3, 4):
            task = {"Dataset": domain, "k": known, "Method": "PA-CBB"}
            task.update(proposed_task_metrics(domain, known, full))
            records.append(task)
            methods = baseline["external"][domain]["known_points"][str(known)]
            for method in METHODS[1:]:
                task = {"Dataset": domain, "k": known, "Method": method}
                task.update(baseline_task_metrics(methods[method]))
                records.append(task)
    return records


def summarize(records: list[dict]) -> tuple[list[dict], dict]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_method[row["Method"]].append(row)

    summary = []
    for method in METHODS:
        row = {"Method": method}
        for key, _, _, _ in METRICS:
            row[key] = float(np.mean([item[key] for item in by_method[method]]))
        summary.append(row)

    metric_wins = {method: 0 for method in METHODS}
    aggregated_ranks = {}
    for key, _, higher, _ in METRICS:
        values = {row["Method"]: row[key] for row in summary}
        ranks = rank_values(values, higher)
        aggregated_ranks[key] = ranks
        best = min(ranks.values())
        for method, rank in ranks.items():
            if rank == best:
                metric_wins[method] += 1

    task_metric_wins = {method: 0 for method in METHODS}
    task_rank_sum = {method: 0.0 for method in METHODS}
    task_pairs = sorted({(row["Dataset"], row["k"]) for row in records})
    for domain, known in task_pairs:
            block = {(r["Method"]): r for r in records if r["Dataset"] == domain and r["k"] == known}
            for key, _, higher, _ in METRICS:
                ranks = rank_values({m: block[m][key] for m in METHODS}, higher)
                best = min(ranks.values())
                for method in METHODS:
                    task_rank_sum[method] += ranks[method]
                    if ranks[method] == best:
                        task_metric_wins[method] += 1

    for row in summary:
        method = row["Method"]
        row["metric_wins"] = metric_wins[method]
        row["task_metric_wins"] = task_metric_wins[method]
        row["mean_rank"] = task_rank_sum[method] / (len(task_pairs) * len(METRICS))
    return summary, aggregated_ranks


def write_csvs(records: list[dict], summary: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    metric_keys = [item[0] for item in METRICS]
    with (TABLES / "super_multimetric_task_detail.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Dataset", "k", "Method"] + metric_keys)
        writer.writeheader()
        writer.writerows(records)
    with (TABLES / "super_multimetric_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Method"] + metric_keys + ["metric_wins", "task_metric_wins", "mean_rank"],
        )
        writer.writeheader()
        writer.writerows(summary)
    by_k_rows = []
    for known in (2, 3, 4):
        block, _ = summarize([row for row in records if row["k"] == known])
        for row in block:
            by_k_rows.append({"k": known, **row})
    with (TABLES / "super_multimetric_by_k_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["k", "Method"] + metric_keys + ["metric_wins", "task_metric_wins", "mean_rank"],
        )
        writer.writeheader()
        writer.writerows(by_k_rows)


def latex_value(value: float, precision: int, rank: float) -> str:
    formatted = f"{value:.{precision}f}"
    if rank == 1:
        return r"\textbf{" + formatted + "}"
    if rank == 2:
        return r"\underline{" + formatted + "}"
    return formatted


def write_latex(records: list[dict]) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Multi-metric leaderboard on the pooled held-out benchmark, reported separately for two, three and four observed anchors. Within each panel, metrics are macro-averaged over five databases, including the two-curve HEA and NIMS audit subsets. F2/F3 are fractions within factors of two/three. Bold and underline denote first and second place, respectively. Basquin-only and bridge-only variants are excluded here and reported only in ablation.}",
        r"\label{tab:super_multimetric}",
        r"\centering\setlength{\tabcolsep}{2.15pt}\renewcommand{\arraystretch}{1.10}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{9}{c}}",
        r"\toprule",
        r"Method & MAE $\downarrow$ & RMSE $\downarrow$ & MedAE $\downarrow$ & P90AE $\downarrow$ & $R^2\uparrow$ & Pearson $r\uparrow$ & F2 $\uparrow$ & F3 $\uparrow$ & Mean rank $\downarrow$ \\",
    ]
    for panel_index, known in enumerate((2, 3, 4)):
        subset = [row for row in records if row["k"] == known]
        summary, ranks = summarize(subset)
        panel_header = (r"\multicolumn{10}{c}{\textbf{(" + chr(97 + panel_index)
                        + ") $k=" + str(known) + r"$ observed anchors}} \\")
        lines += [r"\midrule", panel_header, r"\midrule"]
        for row in summary:
            method = row["Method"]
            values = [latex_value(row[key], precision, ranks[key][method]) for key, _, _, precision in METRICS]
            method_label = r"\textbf{PA-CBB (ours)}" if method == "PA-CBB" else SHORT_METHODS[method]
            tail = [f"{row['mean_rank']:.2f}"]
            if method == "PA-CBB":
                tail = [r"\textbf{" + item + "}" for item in tail]
            lines.append(method_label + " & " + " & ".join(values + tail) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    text = "\n".join(lines) + "\n"
    (TABLES / "super_multimetric_results.tex").write_text(text)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLES / "super_multimetric_results.tex").write_text(text)


def write_json(summary: list[dict]) -> None:
    proposed = next(row for row in summary if row["Method"] == "PA-CBB")
    payload = {
        "aggregation": "unweighted macro-average over 15 dataset-sparsity tasks",
        "metrics": [key for key, _, _, _ in METRICS],
        "metric_count": len(METRICS),
        "task_metric_cell_count": len(DOMAINS) * 3 * len(METRICS),
        "pa_cbb_metric_wins": proposed["metric_wins"],
        "pa_cbb_task_metric_wins": proposed["task_metric_wins"],
        "pa_cbb_mean_rank": proposed["mean_rank"],
        "summary": summary,
    }
    (TABLES / "super_multimetric_summary.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    records = build_records()
    summary, ranks = summarize(records)
    write_csvs(records, summary)
    write_latex(records)
    write_json(summary)
    proposed = next(row for row in summary if row["Method"] == "PA-CBB")
    print(json.dumps({
        "PA-CBB aggregate metric wins": proposed["metric_wins"],
        f"PA-CBB task-metric wins (of {len(DOMAINS) * 3 * len(METRICS)})": proposed["task_metric_wins"],
        "PA-CBB mean rank": proposed["mean_rank"],
    }, indent=2))


if __name__ == "__main__":
    main()
