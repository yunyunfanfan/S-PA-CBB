#!/usr/bin/env python3
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'control7/eval_steps_1_16.json').read_text())
old=json.loads((ROOT/'deep7/eval_steps_1_16.json').read_text())
steps=np.arange(1,17);colors={'2':'#d55e00','3':'#0072b2','4':'#009e73'}

with (ROOT/'control7/steps_1_16.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['steps','known_points','mae_logN','rmse_logN','coverage90','width90_logN','seconds_per_curve_ensemble'])
    for s in steps:
        for k in '234':
            r=p['steps'][str(s)][k];w.writerow([s,k,r['mae_logN'],r['rmse_logN'],r['coverage90'],r['width90_logN'],r['seconds_per_curve_ensemble']])

fig,ax=plt.subplots(1,3,figsize=(15,4.4))
for k in '234':
    mae=[p['steps'][str(s)][k]['mae_logN'] for s in steps]
    omae=[old['steps'][str(s)][k]['mae_logN'] for s in steps]
    cov=[100*p['steps'][str(s)][k]['coverage90'] for s in steps]
    ax[0].plot(steps,mae,'o-',color=colors[k],label=f'{k} known points')
    ax[0].plot(steps,omae,'--',color=colors[k],alpha=.35)
    ax[0].axhline(p['baseline'][k]['mae_logN'],color=colors[k],ls=':',alpha=.7)
    ax[1].plot(steps,cov,'o-',color=colors[k])
etas=[1,2,4,8,12];covs=[];maes=[]
for eta in etas:
    q=json.loads((ROOT/f'control7/val_eta_{eta:.1f}.json').read_text())
    rs=[q['steps']['2'][k] for k in '234'];covs.append(100*np.mean([r['coverage90'] for r in rs]));maes.append(np.mean([r['mae_logN'] for r in rs]))
ax[2].plot(etas,covs,'o-',color='#7b2cbf',label='coverage')
ax2=ax[2].twinx();ax2.plot(etas,maes,'s--',color='#555555',label='MAE')
ax[0].set(title='ControlNet vs direct-conditioning model',xlabel='Sampling steps',ylabel='MAE in log10(N)')
ax[0].legend(frameon=False,fontsize=9);ax[1].axhline(90,color='black',ls='--',lw=1)
ax[1].set(title='Test-set 90% interval coverage (eta=0.5)',xlabel='Sampling steps',ylabel='Coverage (%)',ylim=(-2,100))
ax[2].axhline(90,color='black',ls='--',lw=1);ax[2].set(title='Validation calibration at 2 steps',xlabel='Noise scale eta',ylabel='Coverage (%)',ylim=(0,100));ax2.set_ylabel('Mean MAE in log10(N)')
for a in ax[:2]:a.set_xticks([1,4,8,12,16]);a.grid(alpha=.2)
ax[2].grid(alpha=.2);fig.suptitle('Physics-anchored 1D ControlNet Brownian bridge',fontsize=13);fig.tight_layout()
fig.savefig(ROOT/'control7/controlnet_results.png',dpi=200,bbox_inches='tight');fig.savefig(ROOT/'control7/controlnet_results.pdf',bbox_inches='tight')
