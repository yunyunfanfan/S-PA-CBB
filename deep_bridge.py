#!/usr/bin/env python3
"""Distributed 1D U-Net conditional Brownian bridge for sparse S-N completion.

Designed for 7x Ascend 910B4 with torch_npu, but also supports CPU smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401 - registers the torch.npu backend
except ImportError:
    torch_npu = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


GRID = 32
HASH_DIM = 16
COND_RAW = HASH_DIM * 4 + 6


def setup_device():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    use_npu = hasattr(torch, "npu") and torch.npu.is_available()
    if use_npu:
        torch.npu.set_device(local)
        device = torch.device(f"npu:{local}")
        backend = "hccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world > 1:
        dist.init_process_group(backend=backend)
    return device, rank, local, world


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def hvec(value: str, dim=HASH_DIM):
    out = np.zeros(dim, np.float32)
    value = value or "<missing>"
    for token in str(value).lower().replace("-", " ").split():
        d = hashlib.sha1(token.encode()).digest()
        out[int.from_bytes(d[:2], "little") % dim] += 1.0 if d[2] & 1 else -1.0
    return out


def fit_line(x, y, prior, ridge=0.01):
    xm, ym = x.mean(), y.mean()
    b = (((x-xm)*(y-ym)).sum() + ridge*prior) / (((x-xm)**2).sum()+ridge+1e-8)
    b = float(np.clip(b, -25, -0.05))
    return float(ym-b*xm), b


def load_curves(path: Path, seed: int):
    obj = json.loads(path.read_text())
    curves = []
    for c in obj["curves"]:
        pts = np.asarray([[p[0], p[1]] for p in c["points"] if not p[2] and p[0] > 0 and p[1] > 0], np.float64)
        if len(pts) < 6 or len(np.unique(pts[:, 0])) < 4:
            continue
        x, y = np.log10(pts[:, 0]), np.log10(pts[:, 1])
        if np.ptp(x) < .04:
            continue
        b = np.cov(x, y, bias=True)[0, 1] / (np.var(x)+1e-10)
        if not (-25 < b < -.05):
            continue
        ux = np.unique(x)
        uy = np.asarray([np.median(y[x == v]) for v in ux])
        grid_x = np.linspace(ux.min(), ux.max(), GRID)
        grid_y = np.interp(grid_x, ux, uy)
        cond = c.get("conditions") or {}
        curves.append({"id": c["id"], "name": c.get("name", ""), "x": x.astype(np.float32),
                       "y": y.astype(np.float32), "gx": grid_x.astype(np.float32),
                       "gy": grid_y.astype(np.float32), "family": c.get("material_family", "Unknown"),
                       "am": str(cond.get("am_type", "")), "R": str(cond.get("load_ratio", "")),
                       "test": str(cond.get("test_type", ""))})
    rng = random.Random(seed); rng.shuffle(curves)
    n = len(curves)
    return curves[:int(.6*n)], curves[int(.6*n):int(.8*n)], curves[int(.8*n):]


def make_case(c, obs, prior, ymean, ystd):
    x, y, gx, target = c["x"], c["y"], c["gx"], c["gy"]
    a, b = fit_line(x[obs], y[obs], prior)
    phys = a + b*gx
    resid = (target-phys)/ystd
    omask = np.zeros(GRID, np.float32); ores = np.zeros(GRID, np.float32)
    for j in obs:
        q = int(np.argmin(np.abs(gx-x[j])))
        omask[q] = 1.; ores[q] = (y[j]-phys[q])/ystd
    try: rnum = float(c["R"])
    except Exception: rnum = 0.
    context = np.concatenate([hvec(c["family"]), hvec(c["am"]), hvec(c["R"]), hvec(c["test"]),
        np.asarray([prior/10, b/10, gx.mean()/3, np.ptp(gx), len(obs)/4, rnum], np.float32)])
    channels = np.stack([(phys-ymean)/ystd, ores, omask, np.linspace(-1, 1, GRID, dtype=np.float32)])
    return {"channels": channels.astype(np.float32), "target": resid.astype(np.float32),
            "phys": phys.astype(np.float32), "gx": gx, "context": context.astype(np.float32),
            "omask": omask, "ores": ores, "x": x, "y": y, "obs": np.asarray(obs), "curve": c}


class BridgeDataset(Dataset):
    def __init__(self, curves, prior, ymean, ystd, repeats, seed):
        self.curves, self.prior, self.ymean, self.ystd = curves, prior, ymean, ystd
        self.repeats, self.seed = repeats, seed
    def __len__(self): return len(self.curves)*self.repeats
    def __getitem__(self, idx):
        c = self.curves[idx % len(self.curves)]
        rng = np.random.default_rng(self.seed + idx*1000003 + np.random.randint(0, 100000))
        k = int(rng.integers(2, min(5, len(c["x"])-1)))
        obs = np.sort(rng.choice(len(c["x"]), k, replace=False))
        case = make_case(c, obs, self.prior, self.ymean, self.ystd)
        t = float(rng.uniform(.02, .98))
        noise = rng.normal(size=GRID).astype(np.float32)
        z = (1-t)*case["target"] + math.sqrt(t*(1-t))*noise
        inp = np.concatenate([z[None], case["channels"]], axis=0)
        return torch.from_numpy(inp), torch.tensor(t), torch.from_numpy(case["context"]), torch.from_numpy(case["target"])


def time_embed(t, dim=32):
    half = dim//2
    f = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)*(-math.log(10000)/(half-1)))
    a = t[:, None]*f[None, :]*1000
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class Block(nn.Module):
    def __init__(self, cin, cout, cdim):
        super().__init__()
        self.n1=nn.GroupNorm(4,cin); self.c1=nn.Conv1d(cin,cout,3,padding=1)
        self.n2=nn.GroupNorm(4,cout); self.c2=nn.Conv1d(cout,cout,3,padding=1)
        self.film=nn.Linear(cdim,2*cout); self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,c):
        h=self.c1(F.silu(self.n1(x))); scale,shift=self.film(c).chunk(2,1)
        h=self.n2(h)*(1+scale[:,:,None])+shift[:,:,None]
        return self.c2(F.silu(h))+self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, base=48, cond_dim=128):
        super().__init__()
        self.cond=nn.Sequential(nn.Linear(COND_RAW+32,cond_dim),nn.SiLU(),nn.Linear(cond_dim,cond_dim))
        self.stem=nn.Conv1d(5,base,3,padding=1)
        self.b1=Block(base,base,cond_dim); self.d1=nn.Conv1d(base,base*2,4,2,1)
        self.b2=Block(base*2,base*2,cond_dim); self.d2=nn.Conv1d(base*2,base*4,4,2,1)
        self.mid=Block(base*4,base*4,cond_dim)
        self.u2=Block(base*4+base*2,base*2,cond_dim)
        self.u1=Block(base*2+base,base,cond_dim)
        self.out=nn.Sequential(nn.GroupNorm(4,base),nn.SiLU(),nn.Conv1d(base,1,3,padding=1))
    def forward(self,x,t,rawc):
        c=self.cond(torch.cat([rawc,time_embed(t)],1))
        h1=self.b1(self.stem(x),c); h2=self.b2(self.d1(h1),c); hm=self.mid(self.d2(h2),c)
        u2=F.interpolate(hm,size=h2.shape[-1],mode='linear',align_corners=False)
        u2=self.u2(torch.cat([u2,h2],1),c)
        u1=F.interpolate(u2,size=h1.shape[-1],mode='linear',align_corners=False)
        return self.out(self.u1(torch.cat([u1,h1],1),c)).squeeze(1)


def train(args):
    device,rank,local,world=setup_device()
    torch.manual_seed(args.seed+rank); np.random.seed(args.seed+rank)
    trainc,valc,testc=load_curves(args.data,args.seed)
    slopes=[]; all_y=[]
    for c in trainc:
        _,b=fit_line(c['x'],c['y'],-6,0);slopes.append(b);all_y.extend(c['y'])
    prior=float(np.median(slopes));ymean=float(np.mean(all_y));ystd=float(np.std(all_y))
    ds=BridgeDataset(trainc,prior,ymean,ystd,args.repeats,args.seed+rank*17)
    sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True) if world>1 else None
    dl=DataLoader(ds,batch_size=args.batch,sampler=sampler,shuffle=sampler is None,num_workers=0,drop_last=True)
    model=UNet1D(args.base).to(device)
    if world>1: model=DDP(model,device_ids=[local],broadcast_buffers=False)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    start=time.time(); history=[]
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train(); total=0.;count=0
        for inp,t,ctx,target in dl:
            inp,t,ctx,target=inp.to(device),t.to(device),ctx.to(device),target.to(device)
            pred=model(inp,t,ctx)
            loss=F.mse_loss(pred,target)+.15*F.l1_loss(pred[:,1:]-pred[:,:-1],target[:,1:]-target[:,:-1])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            total+=float(loss.detach());count+=1
        stat=torch.tensor([total,count],device=device)
        if world>1: dist.all_reduce(stat)
        avg=float(stat[0]/stat[1])
        if rank==0 and (epoch%10==0 or epoch==args.epochs-1):
            print(json.dumps({'epoch':epoch+1,'loss':avg,'seconds':round(time.time()-start,1)}),flush=True);history.append([epoch+1,avg])
    if rank==0:
        out=args.outdir;out.mkdir(parents=True,exist_ok=True)
        core=model.module if isinstance(model,DDP) else model
        torch.save({'model':core.state_dict(),'base':args.base,'prior':prior,'ymean':ymean,'ystd':ystd,
                    'seed':args.seed,'split':{'train':len(trainc),'val':len(valc),'test':len(testc)}},out/'model.pt')
        (out/'train_history.json').write_text(json.dumps(history))
        print('saved',out/'model.pt',flush=True)
    if world>1: dist.barrier()
    cleanup()


@torch.no_grad()
def generate(model, inp_channels, context, omask, ores, steps, eta):
    b=inp_channels.shape[0];z=torch.zeros((b,GRID),device=inp_channels.device)
    for i in range(steps):
        tval=1-i/steps;tnext=1-(i+1)/steps
        t=torch.full((b,),tval,device=z.device)
        pred=model(torch.cat([z[:,None],inp_channels],1),t,context)
        z=(1-tnext)*pred
        if eta>0 and tnext>0:
            z=z+eta*math.sqrt(tnext*(1-tnext)/steps)*torch.randn_like(z)
        z=z*(1-omask)+ores*omask
    return z


def evaluate(args):
    device,rank,local,world=setup_device()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    trainc,valc,testc=load_curves(args.data,ck['seed'])
    model=UNet1D(ck['base']).to(device);model.load_state_dict(ck['model']);model.eval()
    local_curves=testc[rank::world]
    results={s:{k:{'err':[],'inside':[],'width':[],'times':[]} for k in (2,3,4)} for s in range(1,17)}
    rng=np.random.default_rng(args.seed+rank)
    for k in (2,3,4):
        cases=[]
        for c in local_curves:
            if len(c['x'])<k+2:continue
            for _ in range(args.masks):
                obs=np.sort(rng.choice(len(c['x']),k,replace=False));cases.append(make_case(c,obs,ck['prior'],ck['ymean'],ck['ystd']))
        for start in range(0,len(cases),args.eval_batch):
            batch=cases[start:start+args.eval_batch]
            ch=torch.tensor(np.stack([x['channels'] for x in batch]),device=device)
            ctx=torch.tensor(np.stack([x['context'] for x in batch]),device=device)
            om=torch.tensor(np.stack([x['omask'] for x in batch]),device=device)
            ore=torch.tensor(np.stack([x['ores'] for x in batch]),device=device)
            for steps in range(1,17):
                draws=[];tic=time.time()
                for e in range(args.ensemble): draws.append(generate(model,ch,ctx,om,ore,steps,args.eta).cpu().numpy())
                elapsed=time.time()-tic;arr=np.stack(draws)
                for j,case in enumerate(batch):
                    pred=case['phys'][None,:]+ck['ystd']*arr[:,j,:]
                    hidden=np.setdiff1d(np.arange(len(case['y'])),case['obs'])
                    vals=np.stack([np.interp(case['x'][hidden],case['gx'],p) for p in pred])
                    truth=case['y'][hidden];mean=vals.mean(0);lo=np.quantile(vals,.05,axis=0);hi=np.quantile(vals,.95,axis=0)
                    r=results[steps][k];r['err'].extend((mean-truth).tolist());r['inside'].extend(((truth>=lo)&(truth<=hi)).tolist());r['width'].extend((hi-lo).tolist())
                results[steps][k]['times'].append(elapsed/len(batch))
    # HCCL does not reliably support Python-object collectives. Reduce compact
    # sufficient statistics on device instead of gathering variable lists.
    stats=torch.zeros((16,3,8),device=device,dtype=torch.float32)
    for s in range(1,17):
        for ki,k in enumerate((2,3,4)):
            r=results[s][k];err=np.asarray(r['err'],dtype=np.float64)
            inside=np.asarray(r['inside'],dtype=np.float64);width=np.asarray(r['width'],dtype=np.float64)
            times=np.asarray(r['times'],dtype=np.float64)
            stats[s-1,ki]=torch.tensor([
                len(err),np.abs(err).sum(),np.square(err).sum(),inside.sum(),
                width.sum(),len(width),times.sum(),len(times)
            ],device=device,dtype=torch.float32)
    if world>1: dist.all_reduce(stats)
    if rank==0:
        report={'model':'conditional 1D U-Net Brownian bridge','devices':world,'steps':{}}
        for s in range(1,17):
            report['steps'][str(s)]={}
            for ki,k in enumerate((2,3,4)):
                n,sa,ss,si,sw,nw,st,nt=stats[s-1,ki].cpu().tolist()
                report['steps'][str(s)][str(k)]={'points':int(n),'mae_logN':sa/n,'rmse_logN':math.sqrt(ss/n),
                    'coverage90':si/n,'width90_logN':sw/nw,'seconds_per_curve_ensemble':st/nt}
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    cleanup()


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('train');p.add_argument('--data',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--epochs',type=int,default=120);p.add_argument('--repeats',type=int,default=16);p.add_argument('--batch',type=int,default=48);p.add_argument('--base',type=int,default=48);p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--seed',type=int,default=20260928)
    p=sub.add_parser('eval');p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--masks',type=int,default=2);p.add_argument('--ensemble',type=int,default=8);p.add_argument('--eta',type=float,default=.35);p.add_argument('--eval-batch',type=int,default=64);p.add_argument('--seed',type=int,default=20260929)
    a=ap.parse_args();train(a) if a.cmd=='train' else evaluate(a)
if __name__=='__main__':main()
