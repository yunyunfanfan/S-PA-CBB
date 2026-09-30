#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent
names=['source_paper','loao_ti6al4v','loao_in718','loao_alsi10mg','loao_316l']
labels=['Source-paper','Ti-6Al-4V OOD','IN718 OOD','AlSi10Mg OOD','316L OOD']
data={n:json.loads((R/f'strict7/{n}/conformal/metrics.json').read_text()) for n in names}
rows=[]
for n,l in zip(names,labels):
    for k in '234':
        q=data[n]['known_points'][k];c=q['pointwise_conformal'];s=q['simultaneous_conformal']
        rows.append([l,k,q['basquin_mae_logN'],c['mae_logN'],100*(1-c['mae_logN']/q['basquin_mae_logN']),q['raw']['coverage90'],c['coverage90'],c['width90_logN'],s['simultaneous_curve_coverage90'],s['width90_logN']])
with (R/'strict7/strict_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['split','known_points','basquin_mae','model_mae','model_gain_percent','raw_coverage90','point_conformal_coverage90','point_width90','simultaneous_curve_coverage90','simultaneous_width90']);w.writerows(rows)

fig,ax=plt.subplots(1,3,figsize=(16,4.6));x=np.arange(len(names));width=.24;colors=['#d55e00','#0072b2','#009e73']
for i,k in enumerate('234'):
    gain=[100*(1-data[n]['known_points'][k]['pointwise_conformal']['mae_logN']/data[n]['known_points'][k]['basquin_mae_logN']) for n in names]
    cov=[100*data[n]['known_points'][k]['pointwise_conformal']['coverage90'] for n in names]
    wid=[data[n]['known_points'][k]['pointwise_conformal']['width90_logN'] for n in names]
    ax[0].bar(x+(i-1)*width,gain,width,color=colors[i],label=f'{k} known')
    ax[1].bar(x+(i-1)*width,cov,width,color=colors[i])
    ax[2].bar(x+(i-1)*width,wid,width,color=colors[i])
ax[0].axhline(0,color='black',lw=1);ax[0].set(title='MAE gain over Basquin',ylabel='Improvement (%)')
ax[1].axhline(90,color='black',ls='--',lw=1);ax[1].set(title='Pointwise conformal coverage',ylabel='Empirical coverage (%)',ylim=(75,100))
ax[2].set(title='Calibrated 90% interval width',ylabel='Width in log10(N)')
for a in ax:
    a.set_xticks(x);a.set_xticklabels(labels,rotation=18,ha='right');a.grid(axis='y',alpha=.2)
ax[0].legend(frameon=False,ncol=3,fontsize=8);fig.suptitle('Strict source and leave-one-alloy-out evaluation (fixed 9 steps)',fontsize=13);fig.tight_layout()
fig.savefig(R/'strict7/strict_results.png',dpi=200,bbox_inches='tight');fig.savefig(R/'strict7/strict_results.pdf',bbox_inches='tight')
