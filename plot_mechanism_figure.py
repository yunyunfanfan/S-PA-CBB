#!/usr/bin/env python3
"""Draw the mechanism figure for physics-anchored S--N curve generation."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication_figures"
PAPER = ROOT / "paper_elsevier_draft" / "figs"
OUT.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

NAVY = "#315C93"
BLUE = "#7399CB"
PALE_BLUE = "#CAD9EC"
ROSE = "#C96F73"
INK = "#252525"
MID = "#747474"
GRID = "#D8DDE5"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 8.6,
        "axes.labelsize": 9.3,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def build_curves():
    """Return a schematic but physically plausible monotone S--N family."""
    x = np.linspace(4.0, 7.25, 260)  # log10 cycles
    basquin = 2.705 - 0.132 * (x - 4.0)

    # A smooth departure from one-slope physics: shoulder followed by a knee.
    residual = (
        0.042 * np.exp(-((x - 4.95) / 0.46) ** 2)
        - 0.048 * np.exp(-((x - 6.10) / 0.43) ** 2)
        + 0.010 * np.tanh((x - 5.55) / 0.23)
    )
    mean = basquin + residual

    anchor_x = np.array([4.35, 5.28, 6.72])
    anchor_y = np.interp(anchor_x, x, mean)

    rng = np.random.default_rng(17)
    zero_at_anchors = np.prod(x[:, None] - anchor_x[None, :], axis=1)
    zero_at_anchors /= np.max(np.abs(zero_at_anchors))
    envelope = 0.008 + 0.018 * np.exp(-((x - 5.95) / 0.85) ** 2)
    samples = []
    for _ in range(28):
        a, b, c = rng.normal(size=3)
        smooth = a + b * np.sin(1.35 * (x - 4.0)) + c * np.cos(2.1 * (x - 4.0))
        smooth /= max(1.0, np.max(np.abs(smooth)))
        candidate = mean + envelope * zero_at_anchors * smooth
        samples.append(candidate)
    return x, basquin, mean, residual, anchor_x, anchor_y, np.asarray(samples)


def main():
    x, basquin, mean, residual, anchor_x, anchor_y, samples = build_curves()
    cycles = 10**x
    stress_phys = 10**basquin
    stress_mean = 10**mean
    stress_samples = 10**samples

    fig, ax = plt.subplots(figsize=(3.55, 3.55))

    lo, hi = np.quantile(stress_samples, [0.05, 0.95], axis=0)
    ax.fill_between(cycles, lo, hi, color=PALE_BLUE, alpha=0.72, lw=0, zorder=1)
    for curve in stress_samples[::2]:
        ax.plot(cycles, curve, color=BLUE, alpha=0.18, lw=0.75, zorder=2)

    ax.plot(
        cycles,
        stress_phys,
        color=MID,
        ls="--",
        lw=1.65,
        zorder=3,
        label="Basquin physical state",
    )
    ax.plot(
        cycles,
        stress_mean,
        color=NAVY,
        lw=2.25,
        zorder=5,
        label="Generated complete-curve mean",
    )
    ax.scatter(10**anchor_x, 10**anchor_y, s=30, facecolor=INK, edgecolor="white", lw=0.6,
               zorder=7, label="Sparse fatigue tests")

    # Residual correction at one representative life.
    x_arrow = 5.93
    y0 = 10 ** np.interp(x_arrow, x, basquin)
    y1 = 10 ** np.interp(x_arrow, x, mean)
    ax.annotate(
        "",
        xy=(10**x_arrow, y1),
        xytext=(10**x_arrow, y0),
        arrowprops=dict(arrowstyle="<->", color=ROSE, lw=1.5),
        zorder=8,
    )
    ax.text(
        10**6.02,
        np.sqrt(y0 * y1),
        "bridge residual\n" + r"$r_\theta$",
        color=ROSE,
        fontsize=7.5,
        va="center",
        fontweight="bold",
    )

    ax.annotate(
        "curvature /\nmulti-slope\ndeparture",
        xy=(1.35e6, np.interp(1.35e6, cycles, stress_mean)),
        xytext=(2.25e6, 330),
        color=NAVY,
        fontsize=7.3,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.0),
    )

    ax.annotate(
        "conditional curve\ndistribution",
        xy=(2.6e5, hi[np.argmin(np.abs(cycles - 2.6e5))]),
        xytext=(5.0e4, 210),
        color=BLUE,
        fontsize=7.3,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
    )

    equation = (
        r"$\log \sigma_{\mathrm{full}}(N)$"
        r" $=$ $\log \sigma_{\mathrm{Basquin}}(N)$ $+$ $r_\theta(N)$"
        "\n"
        r"$r_\theta \sim p_\theta(r\mid\mathrm{anchors,\ conditions,\ physics})$"
    )
    ax.text(
        0.025,
        0.055,
        equation,
        transform=ax.transAxes,
        fontsize=7.4,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.38", fc="white", ec=GRID, lw=0.9, alpha=0.96),
        zorder=10,
    )

    ax.text(
        0.975,
        0.955,
        "Conditions\nalloy | process | $R$ | metadata",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.34", fc="#FBFAF8", ec=ROSE, lw=0.9),
        zorder=10,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 10**7.25)
    ax.set_ylim(135, 590)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$"])
    ax.set_yticks([150, 200, 300, 400, 500])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(r"Cycles to failure, $N_f$")
    ax.set_ylabel(r"Stress amplitude, $\sigma_a$ (MPa)")
    ax.grid(True, which="major", color=GRID, ls="--", lw=0.75, zorder=0)
    ax.grid(False, which="minor")

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=1,
        frameon=False,
        borderaxespad=0,
        handlelength=2.0,
        labelspacing=0.18,
    )

    fig.tight_layout(pad=0.7)
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"figure_00_mechanism.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 500
        fig.savefig(path, **kwargs)
    fig.savefig(PAPER / "mechanism_explanation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
