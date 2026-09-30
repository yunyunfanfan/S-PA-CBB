#!/usr/bin/env python3
"""Public-data smoke test for physics-anchored Brownian S-N completion.

The code intentionally depends only on Python's standard library and NumPy so it
can run on the ARM/Ascend BJTU host without modifying its system Python.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


ROOT = "https://fde.uwaterloo.ca/Fde/Materials/dindex.html"
NUMERIC = re.compile(r"^\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)\s+")
HREF = re.compile(r"href\s*=\s*[\"']?([^\"'\s>#]+)", re.I)


def fetch_public_fde(output: Path, max_pages: int = 1200) -> None:
    """Crawl only HTML files below the public FDE Materials tree."""
    queue = [ROOT]
    seen: set[str] = set()
    curves: list[dict] = []
    fingerprints: set[str] = set()
    while queue and len(seen) < max_pages:
        url = queue.pop(0).split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fatigue-bridge-research/0.1"})
            raw = urllib.request.urlopen(req, timeout=25).read().decode("latin1", "replace")
        except Exception as exc:
            print(f"WARN fetch {url}: {exc}")
            continue

        low_url = url.lower()
        is_derived = any(s in low_url for s in ("_fitted", "_fc.", "calculator", "merged", "compare"))
        if not is_derived:
            curve = parse_curve(raw, url)
            if curve and len(curve["points"]) >= 6:
                fp = hashlib.sha1(json.dumps(curve["points"], sort_keys=True).encode()).hexdigest()
                if fp not in fingerprints:
                    fingerprints.add(fp)
                    curves.append(curve)

        for href in HREF.findall(raw):
            nxt = urllib.parse.urljoin(url, html.unescape(href)).split("#", 1)[0]
            p = urllib.parse.urlparse(nxt)
            if p.netloc != "fde.uwaterloo.ca" or not p.path.startswith("/Fde/Materials/"):
                continue
            if not p.path.lower().endswith((".html", ".htm")):
                continue
            if nxt not in seen and nxt not in queue:
                queue.append(nxt)
        if len(seen) % 100 == 0:
            print(f"crawled={len(seen)} queued={len(queue)} usable_curves={len(curves)}")
        time.sleep(0.01)

    payload = {
        "source": ROOT,
        "license": "GNU GPL (as stated in each FDE data file)",
        "pages_crawled": len(seen),
        "curves": curves,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    npts = sum(len(c["points"]) for c in curves)
    nr = sum(sum(p[2] for p in c["points"]) for c in curves)
    print(f"saved {len(curves)} curves, {npts} points ({nr} run-outs) -> {output}")


def parse_curve(text: str, url: str) -> dict | None:
    stress_unit = "mpa"
    m = re.search(r"#\s*Stress_units\s*=\s*([^\s#<]+)", text, re.I)
    if m:
        stress_unit = m.group(1).lower().rstrip(".")
    factor = 6.894757293 if stress_unit in {"ksi", "kpsi"} else 1.0
    name = ""
    names = re.findall(r"#\s*NAME\s*=\s*([^\r\n#<]+)", text, re.I)
    if names:
        name = " / ".join(x.strip() for x in names[:2])

    points: list[list[float | int]] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") or not NUMERIC.match(line):
            continue
        body = line.split("#", 1)[0]
        vals = re.findall(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?", body)
        if len(vals) < 3:
            continue
        try:
            reversals, stress = float(vals[1]), float(vals[2]) * factor
        except ValueError:
            continue
        if reversals <= 1 or stress <= 0 or stress > 10000:
            continue
        runout = int("runout" in line.lower() or "run-out" in line.lower())
        points.append([round(stress, 9), round(reversals / 2.0, 9), runout])
    if len(points) < 6:
        return None
    return {"id": url, "name": name, "stress_unit_original": stress_unit, "points": points}


def linfit(x: np.ndarray, y: np.ndarray, prior_slope: float, ridge: float) -> tuple[float, float]:
    xm, ym = float(x.mean()), float(y.mean())
    sxx = float(np.sum((x - xm) ** 2))
    b = (float(np.sum((x - xm) * (y - ym))) + ridge * prior_slope) / (sxx + ridge + 1e-12)
    b = float(np.clip(b, -25.0, -0.05))
    return ym - b * xm, b


def full_slope(curve: dict) -> float | None:
    pts = [p for p in curve["points"] if not p[2]]
    if len(pts) < 5:
        return None
    x = np.log10([p[0] for p in pts])
    y = np.log10([p[1] for p in pts])
    if np.ptp(x) < 0.04:
        return None
    b = float(np.sum((x - x.mean()) * (y - y.mean())) / (np.sum((x - x.mean()) ** 2) + 1e-12))
    return b if -25 < b < -0.05 else None


def clean_curves(payload: dict) -> list[dict]:
    out = []
    for c in payload["curves"]:
        pts = [p for p in c["points"] if not p[2]]
        if len(pts) < 6 or len({p[0] for p in pts}) < 4:
            continue
        c = dict(c)
        c["fail_points"] = pts
        if not c.get("material_family"):
            path = urllib.parse.urlparse(c["id"]).path.split("/Materials/", 1)[-1]
            c["material_family"] = path.split("/", 1)[0] if "/" in path else "Other"
        cond = c.get("conditions") or {}
        family = str(c["material_family"])
        am_type = str(cond.get("am_type") or "").strip()
        load_ratio = str(cond.get("load_ratio") or "").strip()
        test_type = str(cond.get("test_type") or "").strip()
        keys = []
        if am_type and load_ratio and test_type:
            keys.append(f"family={family}|am={am_type}|R={load_ratio}|test={test_type}")
        if am_type and load_ratio:
            keys.append(f"family={family}|am={am_type}|R={load_ratio}")
        if am_type:
            keys.append(f"family={family}|am={am_type}")
        keys.append(f"family={family}")
        c["condition_keys"] = keys
        if full_slope(c) is not None:
            out.append(c)
    return out


def _residual_statistics(curves: list[dict], prior_slope: float) -> dict:
    increments, residuals = [], []
    bins = [[] for _ in range(9)]
    spans = []
    for c in curves:
        pts = c["fail_points"]
        x = np.log10([p[0] for p in pts])
        y = np.log10([p[1] for p in pts])
        a, b = linfit(x, y, prior_slope, 0.0)
        u = (x - x.min()) / (np.ptp(x) + 1e-12)
        spans.append(float(np.ptp(x)))
        r = y - (a + b * x)
        order = np.argsort(u)
        us, rs = u[order], r[order]
        for i in range(len(us) - 1):
            du = us[i + 1] - us[i]
            if du > 1e-5:
                increments.append((rs[i + 1] - rs[i]) ** 2 / du)
        residuals.extend(r.tolist())
        for ui, ri in zip(u, r):
            bins[min(8, int(ui * 9))].append(float(ri))
    q = float(np.median(increments)) if increments else 0.05
    noise = float(np.median(np.abs(residuals)) / 0.67449) if residuals else 0.1
    trend = [float(np.median(v)) if v else 0.0 for v in bins]
    return {"q": max(q, 1e-4), "noise": max(noise, 0.02), "trend": trend,
            "median_span": float(np.median(spans)) if spans else 0.3,
            "n_curves": len(curves)}


def learn_conditioner(curves: list[dict], prior_slope: float) -> dict:
    """Learn hierarchical residual priors from shared material/test conditions."""
    model = {"global": _residual_statistics(curves, prior_slope), "groups": {}}
    keys = sorted({key for c in curves for key in c["condition_keys"]})
    for key in keys:
        subset = [c for c in curves if key in c["condition_keys"]]
        if len(subset) >= 5:
            model["groups"][key] = _residual_statistics(subset, prior_slope)
    return model


def conditioned_prior(conditioner: dict, condition_keys: list[str], u: np.ndarray,
                      fitted_slope: float, prior_slope: float, span: float):
    selected = next((key for key in condition_keys if key in conditioner["groups"]), "global")
    stats = conditioner["groups"].get(selected, conditioner["global"])
    trend = np.interp(u, np.linspace(1/18, 17/18, 9), stats["trend"],
                      left=stats["trend"][0], right=stats["trend"][-1])
    slope_factor = np.clip(abs(fitted_slope) / (abs(prior_slope) + 1e-12), 0.6, 1.8)
    span_factor = np.clip(span / (stats["median_span"] + 1e-12), 0.7, 1.5)
    q = stats["q"] * math.sqrt(float(slope_factor * span_factor))
    return trend, float(q), float(stats["noise"]), stats["n_curves"], selected


def kernel(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Brownian motion begun just outside the modelled interval. Conditioning
    # this process on sparse observations creates Brownian-bridge segments.
    return np.minimum(u[:, None] + 0.05, v[None, :] + 0.05)


def predict_case(points: list, obs_idx: np.ndarray, prior_slope: float, ridge: float,
                 conditioner: dict, condition_keys: list[str], scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    x = np.log10([p[0] for p in points])
    y = np.log10([p[1] for p in points])
    u = (x - x.min()) / (np.ptp(x) + 1e-12)
    xo, yo, uo = x[obs_idx], y[obs_idx], u[obs_idx]
    a, b = linfit(xo, yo, prior_slope, ridge)
    base = a + b * x
    trend, q, noise, group_n, selected = conditioned_prior(
        conditioner, condition_keys, u, b, prior_slope, float(np.ptp(x)))
    conditional_source = base + trend
    residual_obs = yo - conditional_source[obs_idx]
    koo = kernel(uo, uo) + np.eye(len(uo)) * 1e-9
    ko = kernel(u, uo)
    alpha = np.linalg.solve(koo, residual_obs)
    mean = conditional_source + ko @ alpha
    cond = kernel(u, u) - ko @ np.linalg.solve(koo, ko.T)
    var = scale * (q * np.maximum(np.diag(cond), 0.0) + noise ** 2)
    var[obs_idx] = 0.0
    mean[obs_idx] = y[obs_idx]
    used = {"condition_group": selected, "condition_training_curves": group_n,
            "fitted_basquin_slope": b, "conditioned_q": q, "conditioned_noise": noise}
    return y, base, mean, np.maximum(var, 0.0), used


def masks(curves: list[dict], k: int, repeats: int, seed: int):
    rng = random.Random(seed + k * 1009)
    for c in curves:
        n = len(c["fail_points"])
        if n < k + 2:
            continue
        for _ in range(repeats):
            yield c, np.array(sorted(rng.sample(range(n), k)), dtype=int)


def score(curves: list[dict], k: int, repeats: int, seed: int, prior: float,
          ridge: float, conditioner: dict, scale: float) -> dict:
    err_base, err_bb, zvals, widths, exact = [], [], [], [], []
    cases = 0
    for c, obs in masks(curves, k, repeats, seed):
        y, base, mean, var, _ = predict_case(c["fail_points"], obs, prior, ridge, conditioner,
                                             c["condition_keys"], scale)
        hidden = np.setdiff1d(np.arange(len(y)), obs)
        sd = np.sqrt(var[hidden])
        err_base.extend((base[hidden] - y[hidden]).tolist())
        err_bb.extend((mean[hidden] - y[hidden]).tolist())
        zvals.extend((np.abs(mean[hidden] - y[hidden]) <= 1.6448536 * sd).tolist())
        widths.extend((2 * 1.6448536 * sd).tolist())
        exact.extend(np.abs(mean[obs] - y[obs]).tolist())
        cases += 1
    eb, eg = np.asarray(err_base), np.asarray(err_bb)
    return {
        "k": k, "cases": cases, "heldout_points": int(len(eg)),
        "basquin_mae_logN": float(np.mean(np.abs(eb))),
        "basquin_rmse_logN": float(np.sqrt(np.mean(eb ** 2))),
        "bridge_mean_mae_logN": float(np.mean(np.abs(eg))),
        "bridge_mean_rmse_logN": float(np.sqrt(np.mean(eg ** 2))),
        "bridge_90_coverage": float(np.mean(zvals)),
        "bridge_90_mean_width_logN": float(np.mean(widths)),
        "max_observed_projection_error": float(max(exact) if exact else 0.0),
    }


def choose_hyperparameters(train: list[dict], valid: list[dict], prior: float, conditioner: dict, seed: int):
    ridges = [0.0, 0.002, 0.01, 0.03, 0.1, 0.3]
    ridge_scores = []
    for r in ridges:
        vals = [score(valid, k, 4, seed, prior, r, conditioner, 1.0)["basquin_mae_logN"] for k in (2, 3, 4)]
        ridge_scores.append((float(np.mean(vals)), r))
    ridge = min(ridge_scores)[1]
    scales = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    scale_scores = []
    for s in scales:
        cov = np.mean([score(valid, k, 5, seed + 7, prior, ridge, conditioner, s)["bridge_90_coverage"] for k in (2, 3, 4)])
        scale_scores.append((abs(cov - 0.9), s, float(cov)))
    _, scale, val_cov = min(scale_scores)
    return ridge, scale, val_cov


def write_example(curve: dict, obs: np.ndarray, prior: float, ridge: float, conditioner: dict,
                  scale: float, outdir: Path, seed: int) -> dict:
    y, base, mean, var, used = predict_case(curve["fail_points"], obs, prior, ridge, conditioner,
                                            curve["condition_keys"], scale)
    rng = np.random.default_rng(seed)
    x = np.log10([p[0] for p in curve["fail_points"]])
    u = (x - x.min()) / (np.ptp(x) + 1e-12)
    uo = u[obs]
    ko = kernel(u, uo)
    koo = kernel(uo, uo) + np.eye(len(uo)) * 1e-9
    cov = scale * used["conditioned_q"] * (kernel(u, u) - ko @ np.linalg.solve(koo, ko.T))
    cov = (cov + cov.T) / 2 + np.eye(len(u)) * 1e-10
    # These are latent complete-curve draws. The reported predictive interval
    # additionally contains point-level experimental scatter.
    samples = rng.multivariate_normal(mean, cov, size=20)
    samples[:, obs] = y[obs]
    order = np.argsort(x)
    csv = ["stress_MPa,cycles,observed,basquin_logN,bridge_mean_logN,bridge_sd_logN," + ",".join(f"sample_{i+1}_logN" for i in range(20))]
    for i in order:
        row = [curve["fail_points"][i][0], curve["fail_points"][i][1], int(i in set(obs.tolist())), base[i], mean[i], math.sqrt(var[i])]
        row += samples[:, i].tolist()
        csv.append(",".join(str(v) for v in row))
    (outdir / "example_predictions.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    return used


def run(data: Path, outdir: Path, seed: int) -> None:
    payload = json.loads(data.read_text(encoding="utf-8"))
    curves = clean_curves(payload)
    if len(curves) < 15:
        raise SystemExit(f"Only {len(curves)} usable curves; at least 15 are needed")
    rng = random.Random(seed)
    rng.shuffle(curves)
    n = len(curves)
    train, valid, test = curves[: int(.6*n)], curves[int(.6*n): int(.8*n)], curves[int(.8*n):]
    slopes = [full_slope(c) for c in train]
    prior = float(np.median(slopes))
    conditioner = learn_conditioner(train, prior)
    ridge, scale, val_cov = choose_hyperparameters(train, valid, prior, conditioner, seed)
    metrics = [score(test, k, 12, seed + 99, prior, ridge, conditioner, scale) for k in (2, 3, 4)]
    runouts = sum(sum(p[2] for p in c["points"]) for c in curves)
    is_am2022 = "22337629" in payload["source"] or "AM2022" in payload["source"]
    report = {
        "experiment": "physics-anchored Brownian-bridge sparse S-N completion smoke test",
        "source": payload["source"], "license": payload["license"],
        "seed": seed, "curves_raw": len(payload["curves"]), "curves_usable": n,
        "failure_points_usable": sum(len(c["fail_points"]) for c in curves),
        "runouts_retained_but_not_scored": runouts,
        "split_by_whole_curve": {"train": len(train), "validation": len(valid), "test": len(test)},
        "conditioning": ["sparse S-N observations", "Basquin source curve and fitted slope",
                         "query stress domain", "material family", "AM process type",
                         "load ratio", "test type"],
        "learned": {"prior_basquin_slope": prior,
                    "global_brownian_diffusion": conditioner["global"]["q"],
                    "global_point_noise_logN": conditioner["global"]["noise"],
                    "hierarchical_condition_models": len(conditioner["groups"]), "ridge": ridge,
                    "variance_scale": scale, "validation_90_coverage": val_cov},
        "metrics": metrics,
        "limitations": [
            ("Feasibility smoke test on FatigueData-AM2022; not yet the final neural bridge."
             if is_am2022 else "Feasibility smoke test on public SAE/FDE data, not the final FatigueData-AM2022 corpus."),
            "Run-outs are parsed and counted but censoring likelihood is not yet implemented.",
            "The lightweight Gaussian Brownian posterior tests the generative mechanism; a neural conditional bridge is a later-stage model.",
            ("Available material, AM-process, load-ratio and test-type fields are used hierarchically; missing fields fall back to broader groups."
             if is_am2022 else "Material family is conditioned on, but detailed process/test metadata are unavailable in this FDE subset.")
        ]
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    candidates = [c for c in test if len(c["fail_points"]) >= 8]
    ex = max(candidates or test, key=lambda c: len(c["fail_points"]))
    erng = random.Random(seed + 404)
    obs = np.array(sorted(erng.sample(range(len(ex["fail_points"])), 3)), dtype=int)
    used = write_example(ex, obs, prior, ridge, conditioner, scale, outdir, seed + 505)
    (outdir / "example_metadata.json").write_text(json.dumps({"curve": ex["id"], "name": ex["name"],
        "observed_indices": obs.tolist(), "bridge_conditions": used}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--max-pages", type=int, default=1200)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--outdir", type=Path, required=True)
    r.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()
    if args.command == "fetch":
        fetch_public_fde(args.output, args.max_pages)
    else:
        run(args.data, args.outdir, args.seed)


if __name__ == "__main__":
    main()
