#!/usr/bin/env python3
"""Build the aligned nine-column source benchmark table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS = [
    ("mae_logN", "MAE $\\downarrow$", False),
    ("rmse_logN", "RMSE $\\downarrow$", False),
    ("median_ae_logN", "MedAE $\\downarrow$", False),
    ("p90_ae_logN", "P90AE $\\downarrow$", False),
    ("r2", "$R^2\\uparrow$", True),
    ("pearson_r", "Pearson $r\\uparrow$", True),
    ("factor_2_accuracy", "F2 $\\uparrow$", True),
    ("factor_3_accuracy", "F3 $\\uparrow$", True),
]


def load_rows():
    ours = json.loads((ROOT / "final_protocol/source_3step_calibrated/metrics.json").read_text())["summary"]
    deterministic = json.loads((ROOT / "final_protocol/deterministic/source_baselines.json").read_text())["known_points"]
    direct_path = ROOT / "final_protocol/deterministic/direct_unet.json"
    if not direct_path.exists():
        raise SystemExit(f"waiting for {direct_path}")
    direct = json.loads(direct_path.read_text())["known_points"]
    rows = {}
    for k in (2, 3, 4):
        panel = {"PA-CBB (ours)": ours[str(k)]["point"]}
        panel.update({name: value["point"] for name, value in deterministic[str(k)].items()})
        panel["Direct U-Net"] = direct[str(k)]["point"]
        rows[k] = panel
    return rows


def decorated(value, rank, digits=3):
    text = f"{value:.{digits}f}"
    if rank == 0:
        return rf"\textbf{{{text}}}"
    if rank == 1:
        return rf"\underline{{{text}}}"
    return text


def main():
    rows = load_rows()
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{AM2022 source-paper-held-out comparison under the unified three-step protocol and identical hash-derived sparse masks. All baselines predict directly from sparse observations and metadata without a physical intermediate state. Bold and underline denote first and second place within each panel. F2/F3 are fractions within factors of two/three.}",
        r"\label{tab:main_v2}",
        r"\centering\setlength{\tabcolsep}{2.15pt}\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{9}{c}}",
        r"\toprule",
        "Method & " + " & ".join(label for _, label, _ in METRICS) + r" & Mean rank $\downarrow$ \\",
        r"\midrule",
    ]
    for k in (2, 3, 4):
        panel = rows[k]; names = list(panel)
        rank_by_metric = {}
        mean_ranks = np.zeros(len(names))
        for key, _, high in METRICS:
            values = np.asarray([panel[name][key] for name in names])
            order = np.argsort(-values if high else values)
            ranks = np.empty(len(names), dtype=int); ranks[order] = np.arange(len(names))
            rank_by_metric[key] = ranks; mean_ranks += ranks + 1
        mean_ranks /= len(METRICS)
        lines += [rf"\multicolumn{{10}}{{c}}{{\textbf{{({chr(95+k)}) $k={k}$ observed anchors}}}} \\", r"\midrule"]
        rank_order = np.argsort(mean_ranks); overall = np.empty(len(names), dtype=int); overall[rank_order] = np.arange(len(names))
        for index, name in enumerate(names):
            label = r"\textbf{PA-CBB (ours)}" if name == "PA-CBB (ours)" else name.replace("Polynomial", "Poly.").replace("RandomForest", "RF").replace("ExtraTrees", "ET")
            values = [decorated(panel[name][key], rank_by_metric[key][index]) for key, _, _ in METRICS]
            values.append(decorated(mean_ranks[index], overall[index], digits=2))
            lines.append(label + " & " + " & ".join(values) + r" \\")
        if k != 4:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    out = ROOT / "paper_elsevier_draft/tables/table2_source_benchmark.tex"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
