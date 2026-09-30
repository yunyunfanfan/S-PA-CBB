#!/usr/bin/env python3
"""Create vector mechanism figures for the PA-CBB manuscript."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication_figures"
PAPER = ROOT / "paper_elsevier_draft" / "figs"
OUT.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

NAVY = "#315C93"
BLUE = "#86A9D5"
PALE_BLUE = "#DCE7F3"
LAVENDER = "#8E83B7"
PALE_LAVENDER = "#E7E2F0"
ROSE = "#C96F73"
PALE_ROSE = "#F2DEDE"
YELLOW = "#F2B94B"
PALE_YELLOW = "#FAEBC8"
INK = "#252525"
MID = "#747A84"
GRID = "#D8DDE5"
OFFWHITE = "#FBFAF8"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def box(ax, xy, wh, text, fc, ec, fontsize=7.7, weight="normal", z=3):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        fc=fc, ec=ec, lw=1.0, zorder=z,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight=weight, zorder=z + 1)
    return patch


def arrow(ax, start, end, color=MID, lw=1.1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                                lw=lw, color=color, linestyle=ls, zorder=2))


def save(fig, stem):
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if suffix == "png":
            kwargs["dpi"] = 450
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    fig.savefig(PAPER / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(7.25, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Lane labels and separators.
    lanes = [(0.69, "OFFLINE DATA PREPARATION", NAVY),
             (0.38, "MODEL TRAINING", LAVENDER),
             (0.07, "NEW-CURVE INFERENCE", ROSE)]
    for y, label, color in lanes:
        ax.text(0.008, y + 0.205, label, color=color, fontsize=8.2,
                fontweight="bold", ha="left", va="top")
    ax.plot([0, 1], [0.66, 0.66], color=GRID, lw=0.8)
    ax.plot([0, 1], [0.35, 0.35], color=GRID, lw=0.8)

    # Offline row.
    xs = [0.04, 0.235, 0.43, 0.625, 0.82]
    labels = [
        "Five public\nfatigue databases",
        "Definition-aware\nharmonization",
        "Article + curve\nduplicate audit",
        "Source-disjoint\nsplits",
        "Complete 32-point\nreference curves",
    ]
    for x, lab in zip(xs, labels):
        box(ax, (x, 0.71), (0.14, 0.14), lab, PALE_BLUE, NAVY)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + 0.14, 0.78), (xs[i + 1] - 0.008, 0.78), NAVY)

    # Training row.
    xs2 = [0.11, 0.32, 0.53, 0.74]
    labels2 = [
        "Sparse-mask\nsimulation",
        "Basquin physical\nendpoint",
        "Brownian-bridge\ntransport training",
        "1D U-Net + zero-init\ncontrol pathway",
    ]
    for x, lab in zip(xs2, labels2):
        box(ax, (x, 0.405), (0.155, 0.14), lab, PALE_LAVENDER, LAVENDER)
    for i in range(len(xs2) - 1):
        arrow(ax, (xs2[i] + 0.155, 0.475), (xs2[i + 1] - 0.008, 0.475), LAVENDER)
    arrow(ax, (0.89, 0.71), (0.82, 0.56), NAVY, ls="--")
    arrow(ax, (0.49, 0.71), (0.19, 0.56), NAVY, ls="--")

    # Inference row.
    xs3 = [0.018, 0.181, 0.344, 0.507, 0.670, 0.833]
    labels3 = [
        "2-4 tests\n+ metadata",
        "Physical endpoint\n+ controls",
        "Three-step\nreverse bridge",
        "Complete-curve\nensemble",
        "Conformal\ncalibration",
        "Fatigue metrics\n+ next test",
    ]
    for i, (x, lab) in enumerate(zip(xs3, labels3)):
        fc = PALE_ROSE if i < 4 else PALE_YELLOW
        ec = ROSE if i < 4 else YELLOW
        box(ax, (x, 0.095), (0.138, 0.14), lab, fc, ec, fontsize=6.8)
    for i in range(len(xs3) - 1):
        c = ROSE if i < 3 else YELLOW
        arrow(ax, (xs3[i] + 0.138, 0.165), (xs3[i + 1] - 0.006, 0.165), c)

    ax.text(0.60, 0.30, "Frozen sampler: 3 steps, 32 draws, anchor projection",
            color=MID, fontsize=7.1, ha="center", va="center")
    ax.text(0.5, 0.985, "Physics-anchored conditional generation: data, training and deployment",
            ha="center", va="top", fontsize=10.2, fontweight="bold", color=INK)
    save(fig, "workflow_mechanism")


def architecture_and_bridge():
    fig = plt.figure(figsize=(7.25, 4.45))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.18, 0.13, 1.0], hspace=0.20)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.005, 0.98, "(a)", fontsize=10.5, fontweight="bold", va="top")
    ax.text(0.05, 0.98, "Multi-scale ControlNet-style condition injection",
            fontsize=9.5, fontweight="bold", va="top")

    # Backbone U-Net.
    box(ax, (0.025, 0.61), (0.115, 0.16), "$z_t$ + time\nembedding", PALE_BLUE, NAVY)
    enc_x = [0.19, 0.335, 0.48]
    dec_x = [0.625, 0.77]
    heights = [0.17, 0.15, 0.13]
    for i, x in enumerate(enc_x):
        box(ax, (x, 0.61 - i * 0.055), (0.095, heights[i]), f"Encoder {i+1}\nresidual blocks", PALE_BLUE, NAVY, 6.8)
    for i, x in enumerate(dec_x):
        box(ax, (x, 0.50 + i * 0.065), (0.095, 0.13 + i * 0.025), f"Decoder {2-i}\n+ skip", PALE_BLUE, NAVY, 6.8)
    box(ax, (0.915, 0.61), (0.07, 0.16), r"$\hat\epsilon$, $\hat y_0$" + "\nlog variance", PALE_BLUE, NAVY, 6.5)
    centers = [(0.14, 0.69), (0.19, 0.69), (0.285, 0.675), (0.335, 0.675),
               (0.43, 0.64), (0.48, 0.64), (0.575, 0.565), (0.625, 0.565),
               (0.72, 0.64), (0.77, 0.64), (0.865, 0.69), (0.915, 0.69)]
    for s, e in zip(centers[::2], centers[1::2]):
        arrow(ax, s, e, NAVY)

    # Control branch.
    box(ax, (0.025, 0.10), (0.20, 0.20), "Condition tensor\nanchors | mask | grid\nphysical state | metadata", PALE_ROSE, ROSE, 6.9)
    ctrl_x = [0.29, 0.44, 0.59]
    for i, x in enumerate(ctrl_x):
        box(ax, (x, 0.115), (0.10, 0.14), f"Control\nscale {i+1}", PALE_ROSE, ROSE, 6.9)
        if i == 0:
            arrow(ax, (0.225, 0.20), (x - 0.008, 0.185), ROSE)
        else:
            arrow(ax, (ctrl_x[i-1] + 0.10, 0.185), (x - 0.008, 0.185), ROSE)

    # Zero-convolution injections.
    targets = [(0.235, 0.60), (0.38, 0.56), (0.525, 0.51)]
    for x, target in zip(ctrl_x, targets):
        box(ax, (x + 0.014, 0.345), (0.072, 0.072), "zero\nprojection", PALE_YELLOW, YELLOW, 6.3)
        arrow(ax, (x + 0.05, 0.255), (x + 0.05, 0.337), ROSE)
        arrow(ax, (x + 0.05, 0.417), target, YELLOW)
    ax.text(0.835, 0.19, "Observed anchors are re-imposed\nafter every reverse update",
            ha="center", va="center", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc=OFFWHITE, ec=GRID, lw=0.9))

    # Brownian bridge states.
    title_ax = fig.add_subplot(gs[1])
    title_ax.axis("off")
    title_ax.text(0.005, 0.5, "(b)", transform=title_ax.transAxes, fontsize=10.5,
                  fontweight="bold", va="center")
    title_ax.text(0.05, 0.5, "Reverse bridge: physical endpoint to conditional complete curves",
                  transform=title_ax.transAxes, fontsize=9.5, fontweight="bold", va="center")

    sub = gs[2].subgridspec(1, 4, wspace=0.20)
    x = np.linspace(0, 1, 80)
    phys = 0.88 - 0.55 * x
    true = phys + 0.085 * np.exp(-((x - 0.32) / 0.20) ** 2) - 0.075 * np.exp(-((x - 0.70) / 0.17) ** 2)
    anchors = np.array([0.16, 0.49, 0.86])
    rng = np.random.default_rng(9)
    ts = [1.0, 0.67, 0.33, 0.0]
    names = ["physical endpoint", "noisy intermediate", "denoised intermediate", "complete draw"]
    for i, (t, name) in enumerate(zip(ts, names)):
        a = fig.add_subplot(sub[0, i])
        variance = 0.065 * np.sqrt(max(t * (1 - t), 0))
        state = t * phys + (1 - t) * true
        if 0 < t < 1:
            state = state + variance * np.sin(6 * np.pi * x + rng.uniform(0, 2*np.pi))
        # Exact anchor projection is visualized at each state.
        state_anchor = np.interp(anchors, x, true)
        state = np.interp(x, np.r_[0, anchors, 1], np.r_[state[0], state_anchor, state[-1]]) * 0.22 + state * 0.78
        state[np.searchsorted(x, anchors).clip(max=len(x)-1)] = state_anchor
        a.plot(x, phys, color=MID, ls="--", lw=1.0)
        a.plot(x, true, color=NAVY, lw=0.9, alpha=0.35)
        a.plot(x, state, color=ROSE if i < 3 else NAVY, lw=1.8)
        a.scatter(anchors, state_anchor, s=13, color=INK, zorder=5)
        a.set_xlim(0, 1); a.set_ylim(0.22, 1.02)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"$t={t:g}$\n{name}", fontsize=7.4, pad=3)
        for spine in a.spines.values():
            spine.set_color(GRID); spine.set_linewidth(0.8)
        if i == 0:
            a.set_ylabel("log cycles", fontsize=7.2)
            a.set_xlabel("log stress", fontsize=7.2)
        if i < 3:
            a.annotate("", xy=(1.09, 0.61), xytext=(1.01, 0.61), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="-|>", color=ROSE, lw=1.0), annotation_clip=False)

    save(fig, "control_bridge_mechanism")


if __name__ == "__main__":
    workflow()
    architecture_and_bridge()
