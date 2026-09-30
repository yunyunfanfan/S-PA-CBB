#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results_bjthu"
metrics = json.loads((RES / "metrics.json").read_text())
meta = json.loads((RES / "example_metadata.json").read_text())
with (RES / "example_predictions.csv").open() as f:
    rows = list(csv.DictReader(f))

stress = np.array([float(r["stress_MPa"]) for r in rows])
life = np.log10([float(r["cycles"]) for r in rows])
observed = np.array([int(r["observed"]) for r in rows], dtype=bool)
base = np.array([float(r["basquin_logN"]) for r in rows])
mean = np.array([float(r["bridge_mean_logN"]) for r in rows])
sd = np.array([float(r["bridge_sd_logN"]) for r in rows])
samples = np.array([[float(r[f"sample_{i}_logN"]) for r in rows] for i in range(1, 21)])

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
ax = axes[0]
order = np.argsort(stress)
for s in samples[:12]:
    ax.plot(s[order], stress[order], color="#7b61a8", alpha=.13, lw=1)
ax.fill_betweenx(stress[order], (mean - 1.6448536*sd)[order],
                 (mean + 1.6448536*sd)[order], color="#7b61a8", alpha=.2,
                 label="Brownian 90% interval")
ax.plot(base[order], stress[order], "--", color="#d97a2b", lw=2, label="Sparse Basquin")
ax.plot(mean[order], stress[order], color="#5b3c88", lw=2.2, label="Bridge posterior mean")
ax.scatter(life[~observed], stress[~observed], s=24, facecolors="white", edgecolors="#333", label="Held-out experiment")
ax.scatter(life[observed], stress[observed], s=58, marker="D", color="#d62728", zorder=5, label="3 observed points")
ax.set_xlabel(r"$\log_{10}(N_f)$")
ax.set_ylabel("Stress amplitude (MPa)")
ax.set_title("Example: conditionally generated completions")
ax.grid(alpha=.2)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
ks = [m["k"] for m in metrics["metrics"]]
base_mae = [m["basquin_mae_logN"] for m in metrics["metrics"]]
bridge_mae = [m["bridge_mean_mae_logN"] for m in metrics["metrics"]]
x = np.arange(len(ks)); w = .34
ax.bar(x-w/2, base_mae, w, color="#d97a2b", label="Basquin")
ax.bar(x+w/2, bridge_mae, w, color="#5b3c88", label="Bridge mean")
ax.set_xticks(x, [str(k) for k in ks])
ax.set_xlabel("Number of observed S–N points")
ax.set_ylabel(r"Held-out MAE in $\log_{10}(N_f)$")
ax.set_title("Whole-curve test split")
ax.set_ylim(0, max(base_mae) * 1.25)
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False)
for i, m in enumerate(metrics["metrics"]):
    ax.text(i, max(base_mae[i], bridge_mae[i]) + .012,
            f"90% cov. {100*m['bridge_90_coverage']:.1f}%", ha="center", fontsize=8)

dataset_name = "FatigueData-AM2022" if "22337629" in metrics["source"] else "Public SAE/FDE"
fig.suptitle(f"Physics-anchored conditional Brownian S–N completion — {dataset_name}", fontsize=13)
fig.text(.01, .01, f"BJTU server; example: {meta['name'] or Path(meta['curve']).name}", fontsize=8, color="#555")
fig.tight_layout(rect=[0, .04, 1, .95])
fig.savefig(RES / "pilot_summary.png", dpi=190)
fig.savefig(RES / "pilot_summary.pdf")
print(RES / "pilot_summary.png")
