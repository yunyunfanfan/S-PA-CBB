#!/usr/bin/env python3
"""Export fixed-step predictive samples for split-conformal calibration."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
try:
    import torch_npu  # noqa:F401
except ImportError:torch_npu=None
from control_score_bridge import ScoreBase,ScoreControl,sample,split_curves
from control_bridge import make_case
from deep_bridge import setup_device,cleanup

@torch.no_grad()
def main(a):
    device,rank,local,world=setup_device();torch.manual_seed(a.seed+rank);np.random.seed(a.seed+rank)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);tr,va,te,_=split_curves(a.data,ck['seed'],split_ids=ck.get('split_ids'));curves=(va if a.split=='val' else te)[rank::world]
    model=ScoreControl(ScoreBase(ck['base'])).to(device);model.load_state_dict(ck['model']);model.eval();rng=np.random.default_rng(a.seed+rank)
    bags={k:{n:[] for n in ('truth','mean','lo','hi','base','case')} for k in (2,3,4)};case_no=rank*10000000
    for k in (2,3,4):
        cases=[]
        for c in curves:
            if len(c['x'])<k+2:continue
            for _ in range(a.masks):
                q=make_case(c,np.sort(rng.choice(len(c['x']),k,replace=False)),ck['prior'],ck['ymean'],ck['ystd']);q['case_no']=case_no;case_no+=1;cases.append(q)
        for st in range(0,len(cases),a.batch):
            ba=cases[st:st+a.batch];static,ctrl,raw,phys,mask,oy=[torch.tensor(np.stack([q[n] for q in ba]),device=device) for n in ('static','control','raw','phys','mask','oy')]
            draws=np.stack([sample(model,static,ctrl,raw,phys,mask,oy,a.steps,ck['sigma'],a.eta).cpu().numpy() for _ in range(a.ensemble)])
            for j,q in enumerate(ba):
                pred=draws[:,j]*ck['ystd']+ck['ymean'];hidden=np.setdiff1d(np.arange(len(q['y'])),q['obs']);vals=np.stack([np.interp(q['x'][hidden],q['gx'],v) for v in pred]);truth=q['y'][hidden]
                base=np.interp(q['x'][hidden],q['gx'],q['phys']*ck['ystd']+ck['ymean']);b=bags[k]
                b['truth'].extend(truth);b['mean'].extend(vals.mean(0));b['lo'].extend(np.quantile(vals,.05,axis=0));b['hi'].extend(np.quantile(vals,.95,axis=0));b['base'].extend(base);b['case'].extend([q['case_no']]*len(hidden))
    a.outdir.mkdir(parents=True,exist_ok=True);part=a.outdir/f'{a.split}_rank{rank}.npz';np.savez_compressed(part,**{f'k{k}_{n}':np.asarray(v) for k,d in bags.items() for n,v in d.items()})
    if world>1:dist.barrier()
    if rank==0:
        merged={}
        for k in (2,3,4):
            for n in ('truth','mean','lo','hi','base','case'):
                merged[f'k{k}_{n}']=np.concatenate([np.load(a.outdir/f'{a.split}_rank{r}.npz')[f'k{k}_{n}'] for r in range(world)])
        np.savez_compressed(a.outdir/f'{a.split}_predictions.npz',**merged);print(a.outdir/f'{a.split}_predictions.npz',flush=True)
    cleanup()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--split',choices=('val','test'),required=True);p.add_argument('--steps',type=int,default=9);p.add_argument('--eta',type=float,default=0.);p.add_argument('--ensemble',type=int,default=32);p.add_argument('--masks',type=int,default=2);p.add_argument('--batch',type=int,default=64);p.add_argument('--seed',type=int,default=20261001);main(p.parse_args())
