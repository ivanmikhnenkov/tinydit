"""Paired bootstrap 95% CIs over prompts for the pass@1 and pass@32 gains on docs/rl/README.md, from the per-prompt counts
in out/passk/summary.json and out/passk_long/summary.json (CPU only).   .venv/bin/python scripts/rl/passk_ci.py
"""
import json, numpy as np
from math import comb
rng=np.random.default_rng(0)
def passk(c,n,k):
    c=np.asarray(c); return np.mean(1-np.array([comb(n-ci,k)/comb(n,k) if n-ci>=k else 0.0 for ci in c]))
def ci(cb,ct,n,k,B=10000):
    cb=np.asarray(cb); ct=np.asarray(ct); P=len(cb)
    pk=lambda c: 1-np.array([comb(n-ci,k)/comb(n,k) if n-ci>=k else 0.0 for ci in c])
    db=pk(ct)-pk(cb); idx=rng.integers(0,P,(B,P)); d=db[idx].mean(1)
    return db.mean(), np.percentile(d,2.5), np.percentile(d,97.5)
out={}
for f,n in [('out/passk/summary.json',64),('out/passk_long/summary.json',32)]:
    s=json.load(open(f))
    for j,d in s['summary'].items():
        base=d['base']['c_per_prompt']
        for name,v in d.items():
            c=v['c_per_prompt']; row={'p1':passk(c,n,1),'p32':passk(c,n,32)}
            if name!='base':
                for k in (1,32): row[f'd{k}']=ci(base,c,n,k)
            out[(f,j,name)]=row
            print(f.split('/')[1],j,name,'n',n,' '.join(f"{k}={v:.3f}" if not isinstance(v,tuple) else f"{k}={v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]" for k,v in row.items()))
