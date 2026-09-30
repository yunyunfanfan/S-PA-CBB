#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent
p=json.loads((R/'score7/test_eta_0.0.json').read_text());direct=json.loads((R/'deep7/eval_steps_1_16.json').read_text());ctrl=json.loads((R/'control7/eval_steps_1_16.json').read_text())
steps=np.arange(1,17);colors={'2':'#d55e00','3':'#0072b2','4':'#009e73'}
with (R/'score7/steps_1_16.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['steps','known_points','mae_logN','rmse_logN','coverage90','width90_logN','seconds_per_curve_ensemble'])
    for s in steps:
        for k in '234':
            q=p['steps'][str(s)][k];w.writerow([s,k,q['mae_logN'],q['rmse_logN'],q['coverage90'],q['width90_logN'],q['seconds_per_curve_ensemble']])
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
for k in '234':
    y=[p['steps'][str(s)][k]['mae_logN'] for s in steps];cov=[100*p['steps'][str(s)][k]['coverage90'] for s in steps]
    ax[0].plot(steps,y,'o-',color=colors[k],label=f'{k} known points');ax[0].axhline(p['baseline'][k]['mae_logN'],color=colors[k],ls=':',alpha=.7)
    ax[1].plot(steps,cov,'o-',color=colors[k])
models=[]
for name,q in [('Direct condition',direct),('ControlNet x0',ctrl),('ControlNet score',p)]:
    vals=[]
    for k in '234':vals.append(min(q['steps'][str(s)][k]['mae_logN'] for s in steps))
    models.append((name,vals))
x=np.arange(3);width=.24
for i,(name,vals) in enumerate(models):ax[2].bar(x+(i-1)*width,vals,width,label=name)
ax[0].set(title='Held-out point accuracy',xlabel='Sampling steps',ylabel='MAE in log10(N)');ax[0].legend(frameon=False,fontsize=9)
ax[1].axhline(90,color='black',ls='--',lw=1);ax[1].set(title='Generated 90% interval',xlabel='Sampling steps',ylabel='Empirical coverage (%)',ylim=(0,100))
ax[2].set(title='Best MAE across 1-16 steps',xlabel='Known points',ylabel='MAE in log10(N)',xticks=x,xticklabels=['2','3','4']);ax[2].legend(frameon=False,fontsize=8)
for a in ax[:2]:a.set_xticks([1,4,8,12,16]);a.grid(alpha=.2)
ax[2].grid(axis='y',alpha=.2);fig.suptitle('ControlNet Brownian bridge with learned score and variance',fontsize=13);fig.tight_layout()
fig.savefig(R/'score7/score_bridge_results.png',dpi=200,bbox_inches='tight');fig.savefig(R/'score7/score_bridge_results.pdf',bbox_inches='tight')
