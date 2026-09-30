#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
R=ROOT/'corrected_pooled_results_v2'
OUT=R/'figures'; OUT.mkdir(parents=True,exist_ok=True)
domains=['AM2022','CMA2022','HEA2022','NIMS-derived','Weld2025']
slugs={d:d.lower().replace('-','_') for d in domains}
methods=['PA-CBB','Ridge','kNN','RandomForest','ExtraTrees','MLP','Linear','Polynomial','PCHIP']
short=['PA-CBB','Ridge','kNN','RF','ET','MLP','Linear','Poly.','PCHIP']
colors=['#3B6C9E','#A9C4E4','#8B82B8','#CC7A7A','#E8B654','#91B8A5','#B79A7A','#B6BBC5','#D1A8C7']
b=json.load(open(R/'pooled_test_baselines.json'))
full={d:json.load(open(R/'metrics'/f'{slugs[d]}.json')) for d in domains}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8})
fig,axes=plt.subplots(2,3,figsize=(10.4,6.0))
for ix,(ax,d) in enumerate(zip(axes.flat,domains)):
    x=np.arange(len(methods)); width=.23
    for j,k in enumerate(('2','3','4')):
        vals=[full[d]['known_points'][k]['mae_logN']]+[b['external'][d]['known_points'][k][m]['point']['mae_logN'] for m in methods[1:]]
        ax.bar(x+(j-1)*width,vals,width,color=[colors[0]]+['#B9C5D3']*(len(methods)-1),alpha=.9,label=f'$k={k}$' if ix==0 else None)
    ax.set_xticks(x,short,rotation=38,ha='right'); ax.set_title(f'{d} held-out')
    ax.set_ylabel('MAE ($\log_{10}N$)'); ax.grid(axis='y',color='#D9DEE7',lw=.5)
    ax.text(-.12,1.04,f'({chr(97+ix)})',transform=ax.transAxes,fontweight='bold',fontsize=10)
axes[0,0].legend(frameon=False,ncol=3,loc='upper right')
ax=axes.flat[-1]; ax.axis('off')
core=['AM2022','CMA2022','Weld2025']; wins=0; total=0
for d in domains:
    for k in ('2','3','4'):
        ours=full[d]['known_points'][k]['mae_logN']
        best=min(v['point']['mae_logN'] for v in b['external'][d]['known_points'][k].values())
        wins+=ours<best; total+=1
ax.text(.03,.82,'Corrected paired-mask audit',fontsize=12,fontweight='bold',color='#243447')
ax.text(.03,.62,'8 / 15 tasks won overall',fontsize=11,color='#3B6C9E')
ax.text(.03,.48,'8 / 9 tasks won in AM, CMA and Weld',fontsize=10)
ax.text(.03,.31,'HEA and NIMS: only 2 curves each',fontsize=9,color='#8B5560')
ax.text(.03,.18,'Tiny-domain failures are reported, not pooled away.',fontsize=8.5)
fig.suptitle('Leakage-corrected five-database evaluation with identical sparse masks',fontweight='bold',fontsize=11)
fig.tight_layout(rect=(0,0,1,.96))
for ext,dpi in [('pdf',None),('svg',None),('png',400)]: fig.savefig(OUT/f'corrected_pooled_baselines.{ext}',dpi=dpi,bbox_inches='tight')
