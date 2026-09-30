#!/usr/bin/env python3
"""Create source-paper and leave-one-alloy-out manifests."""
import json,random
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'am2022_curves.json';OUT=ROOT/'strict_splits';OUT.mkdir(exist_ok=True)
raw=json.loads(DATA.read_text());curves=[]
for c in raw['curves']:
    pts=np.asarray([[p[0],p[1]] for p in c['points'] if not p[2] and p[0]>0 and p[1]>0],float)
    if len(pts)<6 or len(np.unique(pts[:,0]))<4:continue
    x,y=np.log10(pts[:,0]),np.log10(pts[:,1])
    if np.ptp(x)<.04:continue
    b=np.cov(x,y,bias=True)[0,1]/(np.var(x)+1e-10)
    if -25<b<-.05:curves.append(c)
def source(c):return str(c['metadata'].get('metadata: title','')).strip() or c['id']
def material(c):return str(c['conditions'].get('material','')).strip()

def assign_groups(items,fracs,seed):
    groups=defaultdict(list)
    for c in items:groups[source(c)].append(c['id'])
    rng=random.Random(seed);packs=list(groups.values());rng.shuffle(packs);packs.sort(key=len,reverse=True)
    names=list(fracs);targets={n:fracs[n]*len(items) for n in names};out={n:[] for n in names};counts={n:0 for n in names}
    for pack in packs:
        dest=max(names,key=lambda n:(targets[n]-counts[n])/max(targets[n],1))
        out[dest].extend(pack);counts[dest]+=len(pack)
    return out

src=assign_groups(curves,{'train':.6,'val':.2,'test':.2},20260928)
src.update({'name':'source_paper_split','excluded':[]})
(OUT/'source_paper.json').write_text(json.dumps(src,indent=2))

for alloy in ('Ti-6Al-4V','IN718','AlSi10Mg','316L'):
    test=[c for c in curves if material(c)==alloy];test_sources={source(c) for c in test}
    eligible=[c for c in curves if material(c)!=alloy and source(c) not in test_sources]
    excluded=[c['id'] for c in curves if material(c)!=alloy and source(c) in test_sources]
    tv=assign_groups(eligible,{'train':.8,'val':.2},20260928)
    manifest={'name':'leave_alloy_out','held_out_alloy':alloy,'train':tv['train'],'val':tv['val'],'test':[c['id'] for c in test],'excluded':excluded}
    fn='loao_'+alloy.lower().replace('-','').replace(' ','_')+'.json';(OUT/fn).write_text(json.dumps(manifest,indent=2))

for f in sorted(OUT.glob('*.json')):
    q=json.loads(f.read_text());print(f.name,{k:len(q.get(k,[])) for k in ('train','val','test','excluded')})
