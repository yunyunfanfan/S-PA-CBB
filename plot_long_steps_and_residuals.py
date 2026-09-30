#!/usr/bin/env python3
"""Publication figures and LaTeX tables for the sampler and residual audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


BLUE = "#577FAE"; LIGHT_BLUE = "#AFC4DE"; ROSE = "#CE7D7D"
PURPLE = "#8E83B8"; GOLD = "#EAB94F"; INK = "#20242B"; GRID = "#DDE2EA"
CMAP = LinearSegmentedColormap.from_list("paper", [LIGHT_BLUE, "#F7F4F1", ROSE])


def style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .55,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, root, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(root / f"{name}.{suffix}", dpi=400 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def long_step_figure(data, root):
    rows = data["long_step_summary"]; x = np.array([r["steps"] for r in rows])
    panels = [("hidden_mae", "Hidden-point MAE", BLUE), ("grid_mae", "Grid MAE", PURPLE),
              ("energy_score", "Energy Score", ROSE), ("slope_error", "Slope error", GOLD),
              ("violation_rate", "Monotonic violation rate", LIGHT_BLUE),
              ("seconds_per_curve", "Inference time (s/curve)", INK)]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.0))
    for ax, (metric, title, color) in zip(axes.flat, panels):
        y = np.array([r[metric] for r in rows]); se = np.array([r[f"{metric}_se"] for r in rows])
        ax.plot(x, y, "o-", color=color, lw=1.8, ms=4); ax.fill_between(x, y-se, y+se, color=color, alpha=.18)
        ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Reverse steps (one sample)")
        if metric == "seconds_per_curve": ax.set_yscale("log")
        best = int(np.argmin(y)); ax.scatter([x[best]], [y[best]], s=55, facecolors="white", edgecolors=color, zorder=4)
        ax.annotate(f"best: {x[best]}", (x[best], y[best]), xytext=(4, 6), textcoords="offset points", fontsize=7)
    fig.suptitle("Single-sample long-step audit: accuracy saturates before compute cost",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, root, "long_step_single_sample")


def generative_figure(data, root):
    rows = data["ensemble_summary"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.1))
    flat_axes = axes.ravel()
    panels = [("grid_mae", "Ensemble-mean grid MAE", "min"),
              ("crps", "CRPS", "min"), ("energy_score", "Energy Score", "min"),
              ("coverage90", "Raw 90% coverage", "max"),
              ("best_of_m_grid_mae", "Best-of-M grid MAE", "min"),
              ("diversity_rms", "Within-ensemble diversity", "max")]
    for steps, color in ((3, BLUE), (9, ROSE)):
        subset = [r for r in rows if r["steps"] == steps]
        x = np.array([r["ensemble"] for r in subset])
        for ax, (metric, title, _) in zip(flat_axes, panels):
            y = np.array([r[metric] for r in subset])
            ax.plot(x, y, "o-", color=color, lw=1.7, ms=4, label=f"{steps} steps")
            ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
            ax.set_title(title, fontweight="bold"); ax.set_xlabel("Generated samples")
    flat_axes[3].axhline(.9, color=INK, ls="--", lw=1, label="nominal 0.90")
    flat_axes[0].legend(ncol=2); flat_axes[3].legend(fontsize=7)
    fig.suptitle("What multiple stochastic completions add beyond one curve",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, root, "multiple_generation_value")

    noise = data["noise_summary"]; eta = np.array([r["eta"] for r in noise])
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.8))
    for ax, metric, title, color in zip(axes,
        ("grid_mae", "crps", "coverage90", "diversity_rms"),
        ("Grid MAE", "CRPS", "Raw 90% coverage", "Diversity"),
        (BLUE, PURPLE, ROSE, GOLD)):
        y = np.array([r[metric] for r in noise]); ax.plot(eta, y, "o-", color=color, lw=1.8)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel(r"Transition-noise scale $\eta$")
        if metric == "coverage90": ax.axhline(.9, color=INK, ls="--", lw=1)
    fig.suptitle("Noise ablation: stochastic transitions create diversity but require calibration",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout(); save(fig, root, "noise_scale_ablation")


def residual_figure(data, root):
    rows = data["residual_records"]
    required = np.concatenate([r["required_correction"] for r in rows])
    learned = np.concatenate([r["learned_correction"] for r in rows])
    base = np.concatenate([r["physics_residual"] for r in rows])
    generated = np.concatenate([r["generated_residual"] for r in rows])
    position = np.concatenate([r["grid_position"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(10.7, 6.2))
    ax = axes[0,0]; hb=ax.hexbin(required, learned, gridsize=45, mincnt=1, cmap="Blues", bins="log")
    lim=np.nanpercentile(np.abs(np.r_[required, learned]),99); ax.plot([-lim,lim],[-lim,lim],"--",color=ROSE,lw=1)
    ax.set(xlabel="Correction required from physics", ylabel="Correction learned by bridge", title="Learned residual direction")
    fig.colorbar(hb, ax=ax, fraction=.05, pad=.02, label="log count")

    ax=axes[0,1]; abs_base=np.abs(base); abs_generated=np.abs(generated)
    hb=ax.hexbin(abs_base, abs_generated, gridsize=45, mincnt=1, cmap="Purples", bins="log")
    lim=np.nanpercentile(np.r_[abs_base, abs_generated],99); ax.plot([0,lim],[0,lim],"--",color=INK,lw=1)
    ax.set(xlabel="Absolute Basquin residual",ylabel="Absolute PA-CBB residual",title="Residual contraction")
    fig.colorbar(hb, ax=ax, fraction=.05, pad=.02, label="log count")

    bins=np.linspace(0,1,9); centers=(bins[:-1]+bins[1:])/2
    for values,color,label in ((base,ROSE,"Basquin"),(generated,BLUE,"PA-CBB")):
        means=[]; ses=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            v=np.abs(values[(position>=lo)&(position<hi)]); means.append(v.mean()); ses.append(v.std()/np.sqrt(len(v)))
        axes[0,2].plot(centers,means,"o-",color=color,label=label); axes[0,2].fill_between(centers,np.array(means)-ses,np.array(means)+ses,color=color,alpha=.15)
    axes[0,2].set(xlabel="Normalized stress-grid position",ylabel="Absolute residual",title="Where correction helps"); axes[0,2].legend()

    ks=(2,3,4); x=np.arange(3); width=.34
    p=[np.mean([r["physics_mae"] for r in rows if r["known"]==k]) for k in ks]
    g=[np.mean([r["generated_mae"] for r in rows if r["known"]==k]) for k in ks]
    axes[1,0].bar(x-width/2,p,width,color=ROSE,label="Basquin"); axes[1,0].bar(x+width/2,g,width,color=BLUE,label="PA-CBB")
    axes[1,0].set(xticks=x,xticklabels=[f"k={k}" for k in ks],ylabel="Grid MAE",title="Residual magnitude"); axes[1,0].legend()

    difficulty=np.array([r["physics_mae"] for r in rows]); gain=np.array([r["physics_mae"]-r["generated_mae"] for r in rows]); spread=np.array([r["mean_spread"] for r in rows])
    edges=np.quantile(difficulty,np.linspace(0,1,6)); groups=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        groups.append((difficulty>=lo)&(difficulty<=hi))
    mean_gain=np.array([gain[m].mean() for m in groups]); improve=np.array([100*np.mean(gain[m]>0) for m in groups])
    qx=np.arange(5); axes[1,1].bar(qx,mean_gain,color=[LIGHT_BLUE,LIGHT_BLUE,PURPLE,ROSE,ROSE])
    axes[1,1].axhline(0,color=INK,ls="--",lw=1); axes[1,1].set(xticks=qx,xticklabels=["Q1\neasy","Q2","Q3","Q4","Q5\nhard"],ylabel="Mean MAE reduction",title="Benefit by physics difficulty")
    twin=axes[1,1].twinx(); twin.plot(qx,improve,"o-",color=INK,lw=1.2,ms=3); twin.set_ylabel("Curves improved (%)",color=INK); twin.grid(False)

    sc=axes[1,2].scatter(difficulty,gain,c=spread,cmap=CMAP,s=11,alpha=.55,edgecolors="none")
    axes[1,2].axhline(0,color=INK,ls="--",lw=1); axes[1,2].set(xlabel="Basquin grid MAE",ylabel="MAE reduction",title="Does generation target hard cases?")
    fig.colorbar(sc,ax=axes[1,2],fraction=.05,pad=.02,label="ensemble spread")
    for ax in axes.flat: ax.title.set_fontweight("bold")
    fig.suptitle("Physics-to-generation residual decomposition",fontsize=12,fontweight="bold",y=1.01)
    fig.tight_layout(); save(fig, root, "physics_generation_residuals")


def tables(data, root):
    rows=data["long_step_summary"]
    lines=[r"\begin{table*}[t]",r"\caption{Single-sample long-step ablation. Time is synchronized accelerator inference time per curve.}",r"\label{tab:long_steps}",r"\centering\scriptsize",r"\begin{tabular}{rrrrrrr}",r"\toprule",r"Steps & Hidden MAE & Grid MAE & Energy & Slope error & Viol. rate & Time (s)\\",r"\midrule"]
    for q in rows: lines.append(f"{q['steps']} & {q['hidden_mae']:.3f} & {q['grid_mae']:.3f} & {q['energy_score']:.3f} & {q['slope_error']:.3f} & {q['violation_rate']:.3f} & {q['seconds_per_curve']:.4f}\\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (root/"table_long_steps.tex").write_text("\n".join(lines))

    rows=data["noise_summary"]
    lines=[r"\begin{table*}[t]",r"\caption{Transition-noise ablation at three reverse steps and 32 generated curves.}",r"\label{tab:noise_ablation}",r"\centering\scriptsize",r"\begin{tabular}{rrrrrrrr}",r"\toprule",r"$\eta$ & Grid MAE & CRPS & Energy & Coverage & Width & Diversity & Time (s)\\",r"\midrule"]
    for q in rows: lines.append(f"{q['eta']:.2f} & {q['grid_mae']:.3f} & {q['crps']:.3f} & {q['energy_score']:.3f} & {q['coverage90']:.3f} & {q['width90']:.3f} & {q['diversity_rms']:.3f} & {q['seconds_per_curve']:.4f}\\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (root/"table_noise_ablation.tex").write_text("\n".join(lines))


def main(args):
    style(); data=json.loads(args.input.read_text()); args.output_dir.mkdir(parents=True,exist_ok=True)
    long_step_figure(data,args.output_dir); generative_figure(data,args.output_dir); residual_figure(data,args.output_dir); tables(data,args.output_dir)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); main(p.parse_args())
