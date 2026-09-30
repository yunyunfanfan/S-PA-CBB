#!/usr/bin/env python3
"""Promote validation-gated PA-CBB to the main source-domain model.

The script reuses frozen posterior draws, reconstructs the exact Basquin state
from the stored sparse observations, fits one prospective gate per sparse
budget on validation curves, and opens the test split only for the final audit.
Posterior deviations are preserved; the gate changes only the residual mean.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import shutil
import types
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def install_npu_checkpoint_compatibility():
    names = ("torch_npu", "torch_npu.utils", "torch_npu.utils.storage",
             "torch_npu.npu", "torch_npu.npu.random", "torch_npu.npu._format")
    modules = {}
    for name in names:
        module = types.ModuleType(name); module.__path__ = []
        modules[name] = module; sys.modules.setdefault(name, module)
    modules["torch_npu.utils.storage"]._rebuild_npu_tensor = lambda *a: torch._utils._rebuild_tensor_v2(*a[:6])
    modules["torch_npu.npu._format"].Format = type("Format", (), {"__init__": lambda self, *a, **k: None})


install_npu_checkpoint_compatibility()

from control_bridge import make_case  # noqa: E402
from control_score_bridge import split_curves  # noqa: E402
from downstream_fatigue_strength import invert_log_stress, monotone_decreasing  # noqa: E402
from extended_metrics import curve_energy, empirical_crps, interval_score, summarize_point  # noqa: E402


def strength(gx, curve, target=6.0):
    y = monotone_decreasing(np.asarray(curve, float))
    if y.min() <= target <= y.max():
        return float(10 ** invert_log_stress(gx, y, target))
    return float("nan")


def database_from_id(curve_id):
    if curve_id.startswith("FatigueData-AM2022:"): return "AM2022"
    if curve_id.startswith("CMA2022:"): return "CMA2022"
    if curve_id.startswith("HEA2022:"): return "HEA2022"
    if curve_id.startswith("WELD2025:"): return "Weld2025"
    if curve_id.startswith("NIMS-DERIVED:"): return "NIMS-derived"
    return "Other"


def load_rows(bank_dir):
    rows = {}
    metadata = None
    for path in sorted(bank_dir.glob("pacbb_rank*.pkl.gz")):
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        metadata = payload
        for row in payload["rows"]:
            rows[tuple(row["key"])] = row
    return list(rows.values()), metadata


def enrich(rows, curves, checkpoint):
    by_id = {curve["id"]: curve for curve in curves}
    enriched = []
    for row in rows:
        split, curve_id, known = row["key"]
        curve = by_id[curve_id]
        obs = np.asarray(row["observed"], int)
        case = make_case(curve, obs, checkpoint["prior"], checkpoint["ymean"], checkpoint["ystd"])
        physics = case["phys"] * checkpoint["ystd"] + checkpoint["ymean"]
        draws = np.asarray(row["draws"], float)
        mean = draws.mean(axis=0); residual = mean - physics
        width = float(np.mean(np.quantile(draws, .95, axis=0) - np.quantile(draws, .05, axis=0)))
        correction = float(np.mean(np.abs(residual)))
        endpoint = float(.5 * (abs(residual[0]) + abs(residual[-1])))
        coverage = float(np.ptp(curve["x"][obs]) / max(np.ptp(curve["gx"]), 1e-8))
        roughness = float(np.mean(np.abs(np.diff(residual, n=2))))
        linear = np.polyval(np.polyfit(np.arange(len(mean)), mean, 1), np.arange(len(mean)))
        curvature = float(np.sqrt(np.mean((mean - linear) ** 2)))
        enriched.append(dict(row) | {"split": split, "known": int(known), "curve": curve,
                         "physics": physics, "raw_draws": draws,
                         "features": {"correction": correction, "width": width,
                                      "endpoint": endpoint, "coverage": coverage,
                                      "roughness": roughness, "curvature": curvature}})
    return enriched


def scores(rows):
    get = lambda key: np.asarray([r["features"][key] for r in rows], float)
    c, w, e, p = get("correction"), get("width"), get("endpoint"), get("coverage")
    q, k = get("roughness"), get("curvature"); eps = 1e-4
    return {"residual magnitude": c,
            "residual / spread": c / (w + eps),
            "support-weighted residual": c * np.sqrt(p + .05) / (w + eps),
            "shape-aware residual": c * (1 + q + k) / (w + eps),
            "tail-aware residual": (c + .5 * e) / (w + eps)}


def error_arrays(rows, alpha):
    grid, strength_error = [], []
    for row in rows:
        truth, physics = row["truth"], row["physics"]
        raw_mean = row["raw_draws"].mean(axis=0)
        pred = physics + alpha * (raw_mean - physics)
        grid.append(np.mean(np.abs(pred - truth)))
        truth_s, pred_s = strength(row["gx"], truth), strength(row["gx"], pred)
        strength_error.append(abs(pred_s - truth_s) if math.isfinite(truth_s) and math.isfinite(pred_s) else np.nan)
    return np.asarray(grid), np.asarray(strength_error)


def fit_gate(rows):
    base_grid, base_strength = error_arrays(rows, 0)
    valid_strength = np.isfinite(base_strength)
    bg, bs = base_grid.mean(), base_strength[valid_strength].mean()
    candidates = {a: error_arrays(rows, a) for a in (.25, .5, .75, 1.0)}
    best = {"objective": 2.0, "score": "retain physics", "threshold": float("inf"),
            "alpha": 0.0, "activation": 0.0, "grid_mae": float(bg),
            "strength_mae_mpa": float(bs)}
    for name, value in scores(rows).items():
        for threshold in np.unique(np.quantile(value, np.linspace(0, 1, 81))):
            active = value >= threshold
            for alpha, (alt_grid, alt_strength) in candidates.items():
                grid = np.where(active, alt_grid, base_grid)
                strength_err = np.where(active, alt_strength, base_strength)
                good = np.isfinite(strength_err) & valid_strength
                gm, sm = float(grid.mean()), float(strength_err[good].mean())
                if gm > bg or sm > bs:
                    continue
                objective = gm / bg + sm / bs
                key = (objective, gm, sm, active.mean())
                old = (best["objective"], best["grid_mae"], best["strength_mae_mpa"], best["activation"])
                if key < old:
                    best = {"objective": float(objective), "score": name,
                            "threshold": float(threshold), "alpha": float(alpha),
                            "activation": float(active.mean()), "grid_mae": gm,
                            "strength_mae_mpa": sm}
    return best


def apply_gate(rows, gate):
    value = scores(rows)[gate["score"]]
    active = value >= gate["threshold"]
    for row, use, score in zip(rows, active, value):
        raw = row["raw_draws"]; raw_mean = raw.mean(axis=0)
        selected_mean = row["physics"] + (gate["alpha"] if use else 0.0) * (raw_mean - row["physics"])
        # Preserve learned posterior deviations and cross-stress covariance;
        # the gate only changes the correction of the conditional mean.
        row["draws"] = selected_mean[None, :] + (raw - raw_mean[None, :])
        row["selected_mean"] = selected_mean; row["active"] = bool(use); row["gate_score"] = float(score)
    return active


def summarize(rows, validation_rows):
    output = {}
    for known in (2, 3, 4):
        test = [r for r in rows if r["known"] == known]
        val = [r for r in validation_rows if r["known"] == known]
        truth_points, pred_points, lo_points, hi_points = [], [], [], []
        grid_mae, slope, violation, crps, energy, raw_grid, basquin_grid = [], [], [], [], [], [], []
        strength_errors = []
        for r in test:
            curve, obs, draws = r["curve"], np.asarray(r["observed"], int), r["draws"]
            hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
            values = np.stack([np.interp(curve["x"][hidden], curve["gx"], d) for d in draws])
            truth = curve["y"][hidden]; mean = values.mean(axis=0)
            lo, hi = np.quantile(values, [.05, .95], axis=0)
            truth_points.extend(truth); pred_points.extend(mean); lo_points.extend(lo); hi_points.extend(hi)
            grid_mae.append(np.mean(np.abs(r["selected_mean"] - r["truth"])))
            raw_grid.append(np.mean(np.abs(r["raw_draws"].mean(axis=0) - r["truth"])))
            basquin_grid.append(np.mean(np.abs(r["physics"] - r["truth"])))
            slope.append(abs(np.polyfit(r["gx"], r["selected_mean"], 1)[0] - np.polyfit(r["gx"], r["truth"], 1)[0]))
            violation.append(np.mean(np.diff(draws, axis=1) > 0))
            crps.append(np.mean(empirical_crps(values, truth)))
            energy.append(curve_energy(draws, r["truth"]))
            ts, ps = strength(r["gx"], r["truth"]), strength(r["gx"], r["selected_mean"])
            if math.isfinite(ts) and math.isfinite(ps): strength_errors.append(abs(ts-ps))
        # Validation-only conformal expansion for the translated posterior.
        calibration = []
        for r in val:
            curve, obs, draws = r["curve"], np.asarray(r["observed"], int), r["draws"]
            hidden = np.setdiff1d(np.arange(len(curve["y"])), obs)
            values = np.stack([np.interp(curve["x"][hidden], curve["gx"], d) for d in draws])
            lo, hi = np.quantile(values, [.05, .95], axis=0); truth = curve["y"][hidden]
            calibration.extend(np.maximum(lo-truth, truth-hi))
        q = float(np.quantile(calibration, min(1, math.ceil((len(calibration)+1)*.9)/len(calibration)), method="higher"))
        truth_points, pred_points = np.asarray(truth_points), np.asarray(pred_points)
        lo_points, hi_points = np.asarray(lo_points), np.asarray(hi_points)
        output[str(known)] = {"point": summarize_point(truth_points, pred_points),
                              "grid_mae_logN": float(np.mean(grid_mae)),
                              "raw_pacbb_grid_mae_logN": float(np.mean(raw_grid)),
                              "basquin_grid_mae_logN": float(np.mean(basquin_grid)),
                              "slope_error": float(np.mean(slope)),
                              "monotonic_violation": float(np.mean(violation)),
                              "crps_logN": float(np.mean(crps)),
                              "energy_score": float(np.mean(energy)),
                              "raw_coverage90": float(np.mean((truth_points>=lo_points)&(truth_points<=hi_points))),
                              "raw_width90_logN": float(np.mean(hi_points-lo_points)),
                              "conformal_q": q,
                              "conformal_coverage90": float(np.mean((truth_points>=lo_points-q)&(truth_points<=hi_points+q))),
                              "conformal_width90_logN": float(np.mean(hi_points-lo_points+2*q)),
                              "strength_N1e6_mae_mpa": float(np.mean(strength_errors)),
                              "activation_rate": float(np.mean([r["active"] for r in test])),
                              "curves": len(test)}
    return output


def write_bank(rows, metadata, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = []
    for r in rows:
        clean.append({"key": r["key"], "gx": np.asarray(r["gx"], np.float32),
                      "truth": np.asarray(r["truth"], np.float32),
                      "observed": np.asarray(r["observed"], np.int16),
                      "draws": np.asarray(r["draws"], np.float32)})
    payload = {k: metadata[k] for k in ("steps", "ensemble", "eta", "mask_seed", "draw_seed", "split_seed") if k in metadata}
    payload.update({"method": "S-PA-CBB", "rows": clean})
    with gzip.open(output_dir / "pacbb_rank0.pkl.gz", "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_source_table(summary, output):
    root = Path(__file__).resolve().parent
    deterministic = json.loads((root / "multidataset_results/pooled_test_baselines.json").read_text())["external"]["AM2022"]["known_points"]
    metrics = [("mae_logN", "MAE $\\downarrow$", False), ("rmse_logN", "RMSE $\\downarrow$", False),
               ("median_ae_logN", "MedAE $\\downarrow$", False), ("p90_ae_logN", "P90AE $\\downarrow$", False),
               ("r2", "$R^2\\uparrow$", True), ("pearson_r", "Pearson $r\\uparrow$", True),
               ("factor_2_accuracy", "F2 $\\uparrow$", True), ("factor_3_accuracy", "F3 $\\uparrow$", True)]
    lines = [r"\begin{table*}[t]", r"\caption{AM2022 source-paper-held-out comparison under identical sparse masks. S-PA-CBB is the validation-gated selective correction; raw PA-CBB and Basquin are reserved for ablation. Bold and underline mark first and second place.}",
             r"\label{tab:expanded_reconstruction}", r"\centering\scriptsize\setlength{\tabcolsep}{2.8pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             "Method & " + " & ".join(m[1] for m in metrics) + r" \\", r"\midrule"]
    for known in (2,3,4):
        panel = {"S-PA-CBB (ours)": summary[str(known)]["point"]}
        panel.update({name: row["point"] for name,row in deterministic[str(known)].items()
                      if name != "Ridge"})
        names = list(panel)
        ranks = {}
        for key, _, high in metrics:
            vals = np.asarray([panel[n][key] for n in names]); order=np.argsort(-vals if high else vals)
            rank=np.empty(len(names),int); rank[order]=np.arange(len(names)); ranks[key]=rank
        lines += [rf"\multicolumn{{9}}{{c}}{{\textbf{{$k={known}$ observed tests}}}} \\", r"\midrule"]
        for i,name in enumerate(names):
            label = r"\textbf{S-PA-CBB (ours)}" if name.startswith("S-PA") else name.replace("RandomForest","RF").replace("ExtraTrees","ET").replace("Polynomial","Poly.")
            cells=[]
            for key,_,_ in metrics:
                val=f"{panel[name][key]:.3f}"
                if ranks[key][i]==0: val=rf"\textbf{{{val}}}"
                elif ranks[key][i]==1: val=rf"\underline{{{val}}}"
                cells.append(val)
            lines.append(label+" & "+" & ".join(cells)+r" \\")
        if known != 4: lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    output.write_text("\n".join(lines)+"\n")


def main(args):
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    split_seed=checkpoint.get("split_seed",checkpoint.get("seed",20260928))
    train,val_curves,test_curves,_=split_curves(args.data,split_seed,split_ids=checkpoint.get("split_ids"))
    all_curves=train+val_curves+test_curves
    rows,metadata=load_rows(args.bank_dir)
    enriched=enrich(rows,all_curves,checkpoint)
    validation=[r for r in enriched if r["key"][0]=="validation"]
    test=[r for r in enriched if r["key"][0]=="test"]
    gates={}
    for known in (2,3,4):
        gate=fit_gate([r for r in validation if r["known"]==known]); gates[str(known)]=gate
        apply_gate([r for r in validation if r["known"]==known],gate)
        apply_gate([r for r in test if r["known"]==known],gate)
    summary=summarize(test,validation)
    domains = ("AM2022", "CMA2022", "HEA2022", "Weld2025", "NIMS-derived")
    domain_summary = {}
    for label in domains:
        domain_test = [r for r in test if database_from_id(r["curve"]["id"]) == label]
        domain_val = [r for r in validation if database_from_id(r["curve"]["id"]) == label]
        if domain_test:
            # Tiny audit domains may have no curve in the validation partition.
            # The selective gate is already fitted globally; use the untouched
            # global validation split only for conformal expansion in that case.
            calibration_rows = domain_val if domain_val else validation
            domain_summary[label] = summarize(domain_test, calibration_rows)
            domain_summary[label]["calibration_scope"] = (
                "same-domain validation" if domain_val else "global validation"
            )
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result={"model":"S-PA-CBB selective residual-mean correction","definition":"physics + gate*alpha*mean_residual + posterior_deviation",
            "gate_training":"validation only; one gate per observed-test budget","gates":gates,
            "summary":summary, "domain_summary":domain_summary}
    (args.output_dir/"selective_main_metrics.json").write_text(json.dumps(result,indent=2))
    write_bank(validation+test,metadata,args.output_dir/"bank")
    if args.source_table:
        write_source_table(domain_summary["AM2022"],args.output_dir/"table_expanded_reconstruction.tex")
    args.paper_tables.mkdir(parents=True,exist_ok=True)
    if args.source_table:
        shutil.copy2(args.output_dir/"table_expanded_reconstruction.tex",args.paper_tables/"table_expanded_reconstruction.tex")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=Path("am2022_curves.json"))
    p.add_argument("--checkpoint",type=Path,default=Path("ablation_campaign/full_seed20261020/model/model.pt"))
    p.add_argument("--bank-dir",type=Path,default=Path("final_protocol/downstream_task_screen"))
    p.add_argument("--output-dir",type=Path,default=Path("final_protocol/selective_main"))
    p.add_argument("--paper-tables",type=Path,default=Path("paper_elsevier_draft/tables"))
    p.add_argument("--source-table",action="store_true")
    main(p.parse_args())
