#!/usr/bin/env python3
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20260928
EVAL_SEED = 20260929
PRIOR = -5.570137202006509


def fit_line(x, y, prior, ridge=0.01):
    xm, ym = x.mean(), y.mean()
    b = (((x-xm)*(y-ym)).sum() + ridge*prior) / (((x-xm)**2).sum()+ridge+1e-8)
    b = float(np.clip(b, -25, -0.05))
    return float(ym-b*xm), b


def load_test():
    obj = json.loads((ROOT/'am2022_curves.json').read_text())
    curves = []
    for c in obj['curves']:
        pts = np.asarray([[p[0], p[1]] for p in c['points'] if not p[2] and p[0] > 0 and p[1] > 0], float)
        if len(pts) < 6 or len(np.unique(pts[:, 0])) < 4:
            continue
        x, y = np.log10(pts[:, 0]), np.log10(pts[:, 1])
        if np.ptp(x) < .04:
            continue
        b = np.cov(x, y, bias=True)[0, 1] / (np.var(x)+1e-10)
        if -25 < b < -.05:
            curves.append((x, y))
    random.Random(SEED).shuffle(curves)
    return curves[int(.8*len(curves)):]


def same_mask_baseline(test):
    out = {}
    for k in (2, 3, 4):
        errors = []
        for rank in range(7):
            rng = np.random.default_rng(EVAL_SEED+rank)
            local = test[rank::7]
            # Reproduce the evaluator's RNG: earlier k values consume draws first.
            for kk in (2, 3, 4):
                for x, y in local:
                    if len(x) < kk+2:
                        continue
                    for _ in range(2):
                        obs = np.sort(rng.choice(len(x), kk, replace=False))
                        if kk == k:
                            a, b = fit_line(x[obs], y[obs], PRIOR)
                            hidden = np.setdiff1d(np.arange(len(y)), obs)
                            errors.extend((a+b*x[hidden]-y[hidden]).tolist())
                if kk == k:
                    break
        e = np.asarray(errors)
        out[str(k)] = {'points': len(e), 'mae_logN': float(np.mean(np.abs(e))),
                       'rmse_logN': float(np.sqrt(np.mean(e**2)))}
    return out


def main():
    report = json.loads((ROOT/'deep7/eval_steps_1_16.json').read_text())
    baseline = same_mask_baseline(load_test())
    (ROOT/'deep7/basquin_same_masks.json').write_text(json.dumps(baseline, indent=2))

    with (ROOT/'deep7/steps_1_16.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['steps','known_points','mae_logN','rmse_logN','coverage90','width90_logN','seconds_per_curve_ensemble'])
        for s in range(1, 17):
            for k in (2, 3, 4):
                r = report['steps'][str(s)][str(k)]
                w.writerow([s, k, r['mae_logN'], r['rmse_logN'], r['coverage90'], r['width90_logN'], r['seconds_per_curve_ensemble']])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {2:'#d55e00', 3:'#0072b2', 4:'#009e73'}
    steps = np.arange(1, 17)
    for k in (2, 3, 4):
        rs = [report['steps'][str(s)][str(k)] for s in steps]
        axes[0].plot(steps, [r['mae_logN'] for r in rs], 'o-', color=colors[k], label=f'{k} known points')
        axes[0].axhline(baseline[str(k)]['mae_logN'], color=colors[k], ls='--', alpha=.55)
        axes[1].plot(steps, [100*r['coverage90'] for r in rs], 'o-', color=colors[k])
        axes[2].plot(steps, [1000*r['seconds_per_curve_ensemble'] for r in rs], 'o-', color=colors[k])
    axes[0].set(title='Held-out point error', xlabel='Sampling steps', ylabel='MAE in log10(N)')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].axhline(90, color='black', ls='--', lw=1, label='nominal 90%')
    axes[1].set(title='Generative interval calibration', xlabel='Sampling steps', ylabel='Empirical coverage (%)', ylim=(-2, 100))
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].set(title='Inference cost (8 samples)', xlabel='Sampling steps', ylabel='ms per curve')
    for ax in axes:
        ax.grid(alpha=.2); ax.set_xticks([1,4,8,12,16])
    fig.suptitle('Conditional 1D U-Net Brownian bridge on FatigueData-AM2022', fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT/'deep7/steps_1_16.png', dpi=200, bbox_inches='tight')
    fig.savefig(ROOT/'deep7/steps_1_16.pdf', bbox_inches='tight')
    print(json.dumps({'baseline': baseline}, indent=2))


if __name__ == '__main__':
    main()
