#!/usr/bin/env python3
"""ControlNet Brownian bridge with noise, clean-curve and learned-variance heads."""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu=None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler

from deep_bridge import GRID,fit_line,load_curves,setup_device,cleanup
from control_bridge import make_case,BaseUNet,ControlEncoder


def split_curves(data,seed,manifest=None,split_ids=None):
    a,b,c=load_curves(data,seed);allc=a+b+c;byid={q['id']:q for q in allc}
    if split_ids is None and manifest is not None:
        m=json.loads(Path(manifest).read_text());split_ids={k:m[k] for k in ('train','val','test')}
    if split_ids is None:return a,b,c,None
    return tuple([[byid[i] for i in split_ids[k] if i in byid] for k in ('train','val','test')])+ (split_ids,)


class ScoreDataset(Dataset):
    def __init__(self,curves,prior,ymean,ystd,repeats,seed,sigma):
        self.curves,self.prior,self.ymean,self.ystd=curves,prior,ymean,ystd
        self.repeats,self.seed,self.sigma=repeats,seed,sigma
    def __len__(self):return len(self.curves)*self.repeats
    def __getitem__(self,idx):
        c=self.curves[idx%len(self.curves)];rng=np.random.default_rng(self.seed+idx*1000003+np.random.randint(0,100000))
        k=int(rng.integers(2,min(5,len(c['x'])-1)));obs=np.sort(rng.choice(len(c['x']),k,replace=False));q=make_case(c,obs,self.prior,self.ymean,self.ystd)
        t=float(rng.uniform(.02,.98));eps=rng.normal(size=GRID).astype(np.float32);st=self.sigma*math.sqrt(t*(1-t))
        z=(1-t)*q['target']+t*q['phys']+st*eps
        vals=(np.concatenate([z[None],q['static']]),q['control'],q['raw'],q['target'],eps,q['phys'],q['mask'],q['oy'])
        return tuple(torch.from_numpy(v.astype(np.float32)) for v in vals)+(torch.tensor(t,dtype=torch.float32),)


class ScoreBase(BaseUNet):
    def __init__(self,base=48):
        super().__init__(base);self.out=nn.Sequential(nn.GroupNorm(4,base),nn.SiLU(),nn.Conv1d(base,3,3,padding=1))


class ScoreControl(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.control=ControlEncoder(backbone.base);self.control.clone_encoder(backbone)
        for p in self.backbone.parameters():p.requires_grad=False
    def forward(self,x,ctrl,t,raw):return self.backbone(x,t,self.control(ctrl,t,raw))


def bridge_loss(out,target,eps,t,phys,sigma):
    ep,x0,lv=out[:,0],out[:,1],out[:,2].clamp(-5,3)
    # Heteroscedastic Gaussian score loss plus an auxiliary clean-curve head.
    nll=.5*((ep-eps).square()*torch.exp(-lv)+lv).mean()
    clean=F.mse_loss(x0,target)
    smooth=F.l1_loss(x0[:,1:]-x0[:,:-1],target[:,1:]-target[:,:-1])
    # Consistency between the two heads, avoiding division near the terminal endpoint.
    st=sigma*torch.sqrt(t*(1-t)).clamp_min(1e-4)
    z=(1-t[:,None])*target+t[:,None]*phys+st[:,None]*eps
    derived=(z-t[:,None]*phys-st[:,None]*ep)/(1-t[:,None]).clamp_min(.05)
    consistency=F.smooth_l1_loss(x0,derived.detach())
    return nll+.6*clean+.1*smooth+.05*consistency+.002*lv.square().mean()


def epochs(model,dl,sampler,opt,n_epochs,device,rank,world,label,sigma):
    hist=[];start=time.time()
    for epc in range(n_epochs):
        if sampler:sampler.set_epoch(epc)
        model.train();tot=0.;n=0
        for main,ctrl,raw,target,eps,phys,mask,oy,t in dl:
            main,ctrl,raw,target,eps,phys,t=[v.to(device) for v in (main,ctrl,raw,target,eps,phys,t)]
            out=model(main,t) if label=='base' else model(main,ctrl,t,raw);loss=bridge_loss(out,target,eps,t,phys,sigma)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.);opt.step();tot+=float(loss.detach());n+=1
        st=torch.tensor([tot,n],device=device);dist.all_reduce(st) if world>1 else None;avg=float(st[0]/st[1])
        if rank==0 and (epc%10==0 or epc==n_epochs-1):
            print(json.dumps({'stage':label,'epoch':epc+1,'loss':avg,'seconds':round(time.time()-start,1)}),flush=True);hist.append([epc+1,avg])
    return hist


def train(args):
    device,rank,local,world=setup_device();torch.manual_seed(args.seed+rank);np.random.seed(args.seed+rank)
    tr,va,te,split_ids=split_curves(args.data,args.seed,args.manifest);sl=[];ys=[]
    for c in tr:_,b=fit_line(c['x'],c['y'],-6,0);sl.append(b);ys.extend(c['y'])
    prior=float(np.median(sl));ymean=float(np.mean(ys));ystd=float(np.std(ys));ds=ScoreDataset(tr,prior,ymean,ystd,args.repeats,args.seed+17*rank,args.sigma)
    sm=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True) if world>1 else None
    dl=DataLoader(ds,batch_size=args.batch,sampler=sm,shuffle=sm is None,num_workers=0,drop_last=True)
    base=ScoreBase(args.base).to(device);bm=DDP(base,device_ids=[local],broadcast_buffers=False) if world>1 else base
    bh=epochs(bm,dl,sm,torch.optim.AdamW(bm.parameters(),lr=args.lr,weight_decay=1e-4),args.base_epochs,device,rank,world,'base',args.sigma)
    base=bm.module if isinstance(bm,DDP) else bm;bridge=ScoreControl(base).to(device);cm=DDP(bridge,device_ids=[local],broadcast_buffers=False) if world>1 else bridge
    params=[p for p in cm.parameters() if p.requires_grad];ch=epochs(cm,dl,sm,torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4),args.control_epochs,device,rank,world,'control',args.sigma)
    if rank==0:
        args.outdir.mkdir(parents=True,exist_ok=True);core=cm.module if isinstance(cm,DDP) else cm
        torch.save({'model':core.state_dict(),'base':args.base,'prior':prior,'ymean':ymean,'ystd':ystd,'sigma':args.sigma,'seed':args.seed,'split':{'train':len(tr),'val':len(va),'test':len(te)},'split_ids':split_ids},args.outdir/'model.pt')
        (args.outdir/'train_history.json').write_text(json.dumps({'base':bh,'control':ch}));print('saved',args.outdir/'model.pt',flush=True)
    if world>1:dist.barrier()
    cleanup()


@torch.no_grad()
def sample(model,static,ctrl,raw,phys,mask,oy,steps,sigma,eta):
    b=len(phys);tmax=.98;z=phys+sigma*math.sqrt(tmax*(1-tmax))*torch.randn_like(phys)
    grid=torch.linspace(tmax,0,steps+1,device=z.device)
    for i in range(steps):
        tval=float(grid[i]);tnext=float(grid[i+1]);t=torch.full((b,),tval,device=z.device)
        out=model(torch.cat([z[:,None],static],1),ctrl,t,raw);ep,x0,lv=out[:,0],out[:,1],out[:,2].clamp(-5,3)
        st=sigma*math.sqrt(max(tval*(1-tval),1e-6));derived=(z-tval*phys-st*ep)/max(1-tval,.05)
        # Direct head stabilizes the terminal region; score-derived x0 dominates later steps.
        alpha=min(1.,(1-tval)/.35);xhat=(1-alpha)*x0+alpha*derived.clamp(-5,5)
        mean=(1-tnext)*xhat+tnext*phys
        if tnext>0:
            sn=sigma*math.sqrt(tnext*(1-tnext));shared=math.sqrt(max(0.,1-eta*eta))*ep
            fresh=eta*torch.randn_like(z)*torch.exp(.5*lv).clamp(.2,3.)
            z=mean+sn*(shared+fresh)
        else:z=mean
        path=(1-tnext)*oy+tnext*phys;z=z*(1-mask)+path*mask
    return z


def evaluate(args):
    device,rank,local,world=setup_device();ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);tr,va,te,_=split_curves(args.data,ck['seed'],split_ids=ck.get('split_ids'))
    model=ScoreControl(ScoreBase(ck['base'])).to(device);model.load_state_dict(ck['model']);model.eval();rng=np.random.default_rng(args.seed+rank);localc=(va if args.split=='val' else te)[rank::world]
    res={s:{k:{'e':[],'inside':[],'width':[],'time':[]} for k in (2,3,4)} for s in range(1,17)};bres={k:[] for k in (2,3,4)}
    for k in (2,3,4):
        cases=[]
        for c in localc:
            if len(c['x'])<k+2:continue
            for _ in range(args.masks):cases.append(make_case(c,np.sort(rng.choice(len(c['x']),k,replace=False)),ck['prior'],ck['ymean'],ck['ystd']))
        for q in cases:
            h=np.setdiff1d(np.arange(len(q['y'])),q['obs']);bp=np.interp(q['x'][h],q['gx'],q['phys']*ck['ystd']+ck['ymean']);bres[k].extend((bp-q['y'][h]).tolist())
        for st in range(0,len(cases),args.eval_batch):
            ba=cases[st:st+args.eval_batch];static,ctrl,raw,phys,mask,oy=[torch.tensor(np.stack([q[n] for q in ba]),device=device) for n in ('static','control','raw','phys','mask','oy')]
            for steps in range(1,17):
                tic=time.time();draws=[sample(model,static,ctrl,raw,phys,mask,oy,steps,ck['sigma'],args.eta).cpu().numpy() for _ in range(args.ensemble)];elapsed=time.time()-tic;arr=np.stack(draws)
                for j,q in enumerate(ba):
                    pred=arr[:,j]*ck['ystd']+ck['ymean'];h=np.setdiff1d(np.arange(len(q['y'])),q['obs']);vals=np.stack([np.interp(q['x'][h],q['gx'],v) for v in pred]);truth=q['y'][h]
                    mean=vals.mean(0);lo=np.quantile(vals,.05,0);hi=np.quantile(vals,.95,0);r=res[steps][k];r['e'].extend((mean-truth).tolist());r['inside'].extend(((truth>=lo)&(truth<=hi)).tolist());r['width'].extend((hi-lo).tolist())
                res[steps][k]['time'].append(elapsed/len(ba))
    stats=torch.zeros((17,3,8),device=device)
    for ki,k in enumerate((2,3,4)):
        e=np.asarray(bres[k]);stats[16,ki,:3]=torch.tensor([len(e),np.abs(e).sum(),np.square(e).sum()],device=device)
        for s in range(1,17):
            r=res[s][k];e=np.asarray(r['e']);inside=np.asarray(r['inside']);w=np.asarray(r['width']);tm=np.asarray(r['time']);stats[s-1,ki]=torch.tensor([len(e),np.abs(e).sum(),np.square(e).sum(),inside.sum(),w.sum(),len(w),tm.sum(),len(tm)],device=device)
    if world>1:dist.all_reduce(stats)
    if rank==0:
        out={'model':'ControlNet Brownian bridge with learned score and variance','devices':world,'split':args.split,'eta':args.eta,'baseline':{},'steps':{}}
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
    p=sp.add_parser('train');p.add_argument('--data',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--base-epochs',type=int,default=60);p.add_argument('--control-epochs',type=int,default=80);p.add_argument('--repeats',type=int,default=16);p.add_argument('--batch',type=int,default=48);p.add_argument('--base',type=int,default=48);p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--sigma',type=float,default=.8);p.add_argument('--seed',type=int,default=20260928)
    p=sp.add_parser('eval');p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=('val','test'),default='test');p.add_argument('--masks',type=int,default=2);p.add_argument('--ensemble',type=int,default=16);p.add_argument('--eta',type=float,default=.5);p.add_argument('--eval-batch',type=int,default=64);p.add_argument('--seed',type=int,default=20260929)
    a=ap.parse_args();train(a) if a.cmd=='train' else evaluate(a)
if __name__=='__main__':main()
