#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import numpy as np

def qhigher(x,alpha):
    level=min(1.,math.ceil((len(x)+1)*(1-alpha))/len(x))
    return float(np.quantile(x,level,method='higher'))

def score(y,lo,hi):
    # Conformalized quantile regression score. It may be negative when the
    # observation is already well inside the uncalibrated interval.
    return np.maximum(lo-y,y-hi)

def metrics(y,m,lo,hi,case):
    inside=(y>=lo)&(y<=hi);width=hi-lo;alpha=.1
    interval_score=width+(2/alpha)*np.maximum(lo-y,0)+(2/alpha)*np.maximum(y-hi,0)
    curve=[]
    for c in np.unique(case):curve.append(np.all(inside[case==c]))
    return {'points':len(y),'mae_logN':float(np.mean(np.abs(m-y))),'rmse_logN':float(np.sqrt(np.mean((m-y)**2))),
            'coverage90':float(np.mean(inside)),'simultaneous_curve_coverage90':float(np.mean(curve)),
            'width90_logN':float(np.mean(width)),'interval_score90':float(np.mean(interval_score))}

def main(a):
    va=np.load(a.val);te=np.load(a.test);out={'method':'split conformal, fixed 9-step sampler','known_points':{}}
    for k in (2,3,4):
        V={n:va[f'k{k}_{n}'] for n in ('truth','mean','lo','hi','base','case')};T={n:te[f'k{k}_{n}'] for n in ('truth','mean','lo','hi','base','case')}
        sv=score(V['truth'],V['lo'],V['hi']);qpoint=qhigher(sv,.1)
        curve_scores=np.asarray([np.max(sv[V['case']==c]) for c in np.unique(V['case'])]);qcurve=qhigher(curve_scores,.1)
        def scaled(q):return T['lo']-q,T['hi']+q
        plo,phi=scaled(qpoint);clo,chi=scaled(qcurve)
        out['known_points'][str(k)]={'basquin_mae_logN':float(np.mean(np.abs(T['base']-T['truth']))),
            'raw':metrics(T['truth'],T['mean'],T['lo'],T['hi'],T['case']),
            'pointwise_conformal_q':qpoint,'pointwise_conformal':metrics(T['truth'],T['mean'],plo,phi,T['case']),
            'simultaneous_conformal_q':qcurve,'simultaneous_conformal':metrics(T['truth'],T['mean'],clo,chi,T['case'])}
    a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--val',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
