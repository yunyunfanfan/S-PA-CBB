#!/usr/bin/env python3
"""Physics-anchored 1D ControlNet Brownian bridge for sparse S-N curves."""
from __future__ import annotations

import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from deep_bridge import GRID, COND_RAW, Block, fit_line, hvec, load_curves, setup_device, cleanup, time_embed


def make_case(c, obs, prior, ymean, ystd):
    x,y,gx,gy=c['x'],c['y'],c['gx'],c['gy']
    a,b=fit_line(x[obs],y[obs],prior);phys=a+b*gx
    target=(gy-ymean)/ystd;physn=(phys-ymean)/ystd
    mask=np.zeros(GRID,np.float32);oy=np.zeros(GRID,np.float32)
    for j in obs:
        q=int(np.argmin(np.abs(gx-x[j])));mask[q]=1.;oy[q]=(y[j]-ymean)/ystd
    try:rnum=float(c['R'])
    except Exception:rnum=0.
    raw=np.concatenate([hvec(c['family']),hvec(c['am']),hvec(c['R']),hvec(c['test']),
        np.asarray([prior/10,b/10,gx.mean()/3,np.ptp(gx),len(obs)/4,rnum],np.float32)])
    stress=np.linspace(-1,1,GRID,dtype=np.float32)
    static=np.stack([oy,mask,stress]);control=np.stack([physn,oy,mask,stress])
    return {'static':static,'control':control,'target':target.astype(np.float32),'phys':physn.astype(np.float32),
            'raw':raw.astype(np.float32),'mask':mask,'oy':oy,'gx':gx,'x':x,'y':y,'obs':np.asarray(obs)}


class CurveDataset(Dataset):
    def __init__(self,curves,prior,ymean,ystd,repeats,seed):
        self.curves,self.prior,self.ymean,self.ystd=curves,prior,ymean,ystd
        self.repeats,self.seed=repeats,seed
    def __len__(self):return len(self.curves)*self.repeats
    def __getitem__(self,idx):
        c=self.curves[idx%len(self.curves)]
        rng=np.random.default_rng(self.seed+idx*1000003+np.random.randint(0,100000))
        k=int(rng.integers(2,min(5,len(c['x'])-1)));obs=np.sort(rng.choice(len(c['x']),k,replace=False))
        q=make_case(c,obs,self.prior,self.ymean,self.ystd);t=float(rng.uniform(.02,.98))
        z=(1-t)*q['target']+t*q['phys']+math.sqrt(t*(1-t))*rng.normal(size=GRID).astype(np.float32)
        return tuple(torch.from_numpy(v) for v in (np.concatenate([z[None],q['static']]),q['control'],q['raw'],q['target']))+(torch.tensor(t),)


class BaseUNet(nn.Module):
    def __init__(self,base=48,cdim=128):
        super().__init__();self.base=base
        self.tcond=nn.Sequential(nn.Linear(32,cdim),nn.SiLU(),nn.Linear(cdim,cdim))
        self.stem=nn.Conv1d(4,base,3,padding=1)
        self.b1=Block(base,base,cdim);self.d1=nn.Conv1d(base,base*2,4,2,1)
        self.b2=Block(base*2,base*2,cdim);self.d2=nn.Conv1d(base*2,base*4,4,2,1)
        self.mid=Block(base*4,base*4,cdim)
        self.u2=Block(base*6,base*2,cdim);self.u1=Block(base*3,base,cdim)
        self.out=nn.Sequential(nn.GroupNorm(4,base),nn.SiLU(),nn.Conv1d(base,1,3,padding=1))
    def forward(self,x,t,adds=None):
        c=self.tcond(time_embed(t));h1=self.b1(self.stem(x),c)
        h2=self.b2(self.d1(h1),c);hm=self.mid(self.d2(h2),c)
        if adds is not None:h1=h1+adds[0];h2=h2+adds[1];hm=hm+adds[2]
        u2=F.interpolate(hm,size=h2.shape[-1],mode='linear',align_corners=False)
        u2=self.u2(torch.cat([u2,h2],1),c)
        u1=F.interpolate(u2,size=h1.shape[-1],mode='linear',align_corners=False)
        return self.out(self.u1(torch.cat([u1,h1],1),c)).squeeze(1)


class ControlEncoder(nn.Module):
    def __init__(self,base=48,cdim=128):
        super().__init__();self.cond=nn.Sequential(nn.Linear(COND_RAW+32,cdim),nn.SiLU(),nn.Linear(cdim,cdim))
        self.stem=nn.Conv1d(4,base,3,padding=1);self.b1=Block(base,base,cdim);self.d1=nn.Conv1d(base,base*2,4,2,1)
        self.b2=Block(base*2,base*2,cdim);self.d2=nn.Conv1d(base*2,base*4,4,2,1);self.mid=Block(base*4,base*4,cdim)
        self.zero1=nn.Conv1d(base,base,1);self.zero2=nn.Conv1d(base*2,base*2,1);self.zerom=nn.Conv1d(base*4,base*4,1)
        for m in (self.zero1,self.zero2,self.zerom):nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def clone_encoder(self,base):
        for name in ('stem','b1','d1','b2','d2','mid'):
            getattr(self,name).load_state_dict(getattr(base,name).state_dict())
    def forward(self,x,t,raw):
        c=self.cond(torch.cat([raw,time_embed(t)],1));h1=self.b1(self.stem(x),c)
        h2=self.b2(self.d1(h1),c);hm=self.mid(self.d2(h2),c)
        return self.zero1(h1),self.zero2(h2),self.zerom(hm)


class ControlBridge(nn.Module):
    def __init__(self,base):
        super().__init__();self.backbone=base;self.control=ControlEncoder(base.base);self.control.clone_encoder(base)
        for p in self.backbone.parameters():p.requires_grad=False
    def forward(self,x,ctrl,t,raw):return self.backbone(x,t,self.control(ctrl,t,raw))


def loss_fn(pred,target):
    return F.mse_loss(pred,target)+.15*F.l1_loss(pred[:,1:]-pred[:,:-1],target[:,1:]-target[:,:-1])


def run_epochs(model,dl,sampler,opt,epochs,device,rank,world,label):
    start=time.time();history=[]
    for ep in range(epochs):
        if sampler:sampler.set_epoch(ep)
        model.train();tot=0.;n=0
        for main,ctrl,raw,target,t in dl:
            main,ctrl,raw,target,t=[v.to(device) for v in (main,ctrl,raw,target,t)]
            pred=model(main,t) if label=='base' else model(main,ctrl,t,raw)
            loss=loss_fn(pred,target);opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.);opt.step();tot+=float(loss.detach());n+=1
        stat=torch.tensor([tot,n],device=device);dist.all_reduce(stat) if world>1 else None;avg=float(stat[0]/stat[1])
        if rank==0 and (ep%10==0 or ep==epochs-1):
            print(json.dumps({'stage':label,'epoch':ep+1,'loss':avg,'seconds':round(time.time()-start,1)}),flush=True);history.append([ep+1,avg])
    return history


def train(args):
    device,rank,local,world=setup_device();torch.manual_seed(args.seed+rank);np.random.seed(args.seed+rank)
    tr,va,te=load_curves(args.data,args.seed);sl=[];ys=[]
    for c in tr:_,b=fit_line(c['x'],c['y'],-6,0);sl.append(b);ys.extend(c['y'])
    prior=float(np.median(sl));ymean=float(np.mean(ys));ystd=float(np.std(ys))
    ds=CurveDataset(tr,prior,ymean,ystd,args.repeats,args.seed+rank*17)
    sm=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True) if world>1 else None
    dl=DataLoader(ds,batch_size=args.batch,sampler=sm,shuffle=sm is None,num_workers=0,drop_last=True)
    base=BaseUNet(args.base).to(device);bm=DDP(base,device_ids=[local],broadcast_buffers=False) if world>1 else base
    bh=run_epochs(bm,dl,sm,torch.optim.AdamW(bm.parameters(),lr=args.lr,weight_decay=1e-4),args.base_epochs,device,rank,world,'base')
    base=bm.module if isinstance(bm,DDP) else bm
    bridge=ControlBridge(base).to(device);cm=DDP(bridge,device_ids=[local],broadcast_buffers=False) if world>1 else bridge
    params=[p for p in cm.parameters() if p.requires_grad]
    ch=run_epochs(cm,dl,sm,torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4),args.control_epochs,device,rank,world,'control')
    if rank==0:
        args.outdir.mkdir(parents=True,exist_ok=True);core=cm.module if isinstance(cm,DDP) else cm
        torch.save({'model':core.state_dict(),'base':args.base,'prior':prior,'ymean':ymean,'ystd':ystd,'seed':args.seed,
                    'split':{'train':len(tr),'val':len(va),'test':len(te)}},args.outdir/'model.pt')
        (args.outdir/'train_history.json').write_text(json.dumps({'base':bh,'control':ch}));print('saved',args.outdir/'model.pt',flush=True)
    if world>1:dist.barrier()
    cleanup()


@torch.no_grad()
def generate(model,static,ctrl,raw,phys,mask,oy,steps,eta):
    z=phys.clone();b=len(z)
    for i in range(steps):
        tval=1-i/steps;tnext=1-(i+1)/steps;t=torch.full((b,),tval,device=z.device)
        x0=model(torch.cat([z[:,None],static],1),ctrl,t,raw)
        z=(1-tnext)*x0+tnext*phys
        if eta and tnext>0:z=z+eta*math.sqrt(tnext*(1-tnext)/steps)*torch.randn_like(z)
        z=z*(1-mask)+oy*mask
    return z


def evaluate(args):
    device,rank,local,world=setup_device();ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    tr,va,te=load_curves(args.data,ck['seed']);base=BaseUNet(ck['base']);model=ControlBridge(base).to(device)
    model.load_state_dict(ck['model']);model.eval();rng=np.random.default_rng(args.seed+rank)
    localc=(va if args.split=='val' else te)[rank::world]
    res={s:{k:{'e':[],'in':[],'w':[],'tm':[]} for k in (2,3,4)} for s in range(1,17)};bres={k:[] for k in (2,3,4)}
    for k in (2,3,4):
        cases=[]
        for c in localc:
            if len(c['x'])<k+2:continue
            for _ in range(args.masks):cases.append(make_case(c,np.sort(rng.choice(len(c['x']),k,replace=False)),ck['prior'],ck['ymean'],ck['ystd']))
        for q in cases:
            hidden=np.setdiff1d(np.arange(len(q['y'])),q['obs']);bp=np.interp(q['x'][hidden],q['gx'],q['phys']*ck['ystd']+ck['ymean']);bres[k].extend((bp-q['y'][hidden]).tolist())
        for st in range(0,len(cases),args.eval_batch):
            ba=cases[st:st+args.eval_batch]
            static,ctrl,raw,phys,mask,oy=[torch.tensor(np.stack([q[n] for q in ba]),device=device) for n in ('static','control','raw','phys','mask','oy')]
            for steps in range(1,17):
                tic=time.time();draws=[generate(model,static,ctrl,raw,phys,mask,oy,steps,args.eta).cpu().numpy() for _ in range(args.ensemble)];elapsed=time.time()-tic;arr=np.stack(draws)
                for j,q in enumerate(ba):
                    pred=arr[:,j]*ck['ystd']+ck['ymean'];hidden=np.setdiff1d(np.arange(len(q['y'])),q['obs'])
                    vals=np.stack([np.interp(q['x'][hidden],q['gx'],p) for p in pred]);truth=q['y'][hidden];mean=vals.mean(0);lo=np.quantile(vals,.05,0);hi=np.quantile(vals,.95,0)
                    r=res[steps][k];r['e'].extend((mean-truth).tolist());r['in'].extend(((truth>=lo)&(truth<=hi)).tolist());r['w'].extend((hi-lo).tolist())
                res[steps][k]['tm'].append(elapsed/len(ba))
    stats=torch.zeros((17,3,8),device=device)
    for ki,k in enumerate((2,3,4)):
        e=np.asarray(bres[k]);stats[16,ki,:3]=torch.tensor([len(e),np.abs(e).sum(),np.square(e).sum()],device=device)
        for s in range(1,17):
            r=res[s][k];e=np.asarray(r['e']);inside=np.asarray(r['in']);w=np.asarray(r['w']);tm=np.asarray(r['tm'])
            stats[s-1,ki]=torch.tensor([len(e),np.abs(e).sum(),np.square(e).sum(),inside.sum(),w.sum(),len(w),tm.sum(),len(tm)],device=device)
    if world>1:dist.all_reduce(stats)
    if rank==0:
        out={'model':'physics-anchored 1D ControlNet Brownian bridge','devices':world,'split':args.split,'eta':args.eta,'baseline':{},'steps':{}}
        for ki,k in enumerate((2,3,4)):
            n,sa,ss=stats[16,ki,:3].cpu().tolist();out['baseline'][str(k)]={'points':int(n),'mae_logN':sa/n,'rmse_logN':math.sqrt(ss/n)}
        for s in range(1,17):
            out['steps'][str(s)]={}
            for ki,k in enumerate((2,3,4)):
                n,sa,ss,si,sw,nw,st,nt=stats[s-1,ki].cpu().tolist();out['steps'][str(s)][str(k)]={'points':int(n),'mae_logN':sa/n,'rmse_logN':math.sqrt(ss/n),'coverage90':si/n,'width90_logN':sw/nw,'seconds_per_curve_ensemble':st/nt}
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
    cleanup()


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('train');p.add_argument('--data',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--base-epochs',type=int,default=60);p.add_argument('--control-epochs',type=int,default=80);p.add_argument('--repeats',type=int,default=16);p.add_argument('--batch',type=int,default=48);p.add_argument('--base',type=int,default=48);p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--seed',type=int,default=20260928)
    p=sp.add_parser('eval');p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=('val','test'),default='test');p.add_argument('--masks',type=int,default=2);p.add_argument('--ensemble',type=int,default=8);p.add_argument('--eta',type=float,default=.5);p.add_argument('--eval-batch',type=int,default=64);p.add_argument('--seed',type=int,default=20260929)
    a=ap.parse_args();train(a) if a.cmd=='train' else evaluate(a)
if __name__=='__main__':main()
