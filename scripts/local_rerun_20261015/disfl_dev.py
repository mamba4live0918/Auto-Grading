"""D1 development on the 24 P clips only (plan section D1)."""
import json, csv, numpy as np, hashlib
import disfl as D, disfl_io as IO
from scipy.stats import spearmanr
import localpaths as LP
U=LP.PC_KEY_CLOUD  # clip windows only
K={k['clip_id']:k for k in json.load(open(U,encoding='utf-8'))}
lab={}
for r in csv.DictReader(open(LP.PC_LABELS,encoding='utf-8-sig')):
    if r['type']!='none': lab.setdefault(r['clip_id'],[]).append((r['type'],float(r['start_s']),float(r['end_s'])))
clips=sorted(K)
ov=lambda x,y: min(x[1],y[1])-max(x[0],y[0])
def match(E,H):
    pairs=sorted(((ov(e,h),i,j) for i,e in enumerate(E) for j,h in enumerate(H)),reverse=True); ue=set();uh=set();m=0
    for o,i,j in pairs:
        if o<=0: break
        if i in ue or j in uh: continue
        ue.add(i);uh.add(j);m+=1
    return m
res=[]
for p in D.GRID:
    tp=nE=nH=anyprob=0; cnt=[];hum=[]
    for c in clips:
        s0=float(K[c]['start_s']); sid=K[c]['sample_id']
        pr,seg,fr=IO.rec(sid)
        ev=[(a-s0,b-s0) for a,b in D.detect(*fr,*p,interval=(s0,s0+30))]
        H=[(a,b) for t,a,b in lab.get(c,[]) if t=='uh']; A=[(a,b) for t,a,b in lab.get(c,[])]
        tp+=match(ev,H); nE+=len(ev); nH+=len(H); anyprob+=sum(1 for e in ev if any(ov(e,h)>0 for h in A))
        cnt.append(len(ev)); hum.append(len(H))
    rec_=tp/nH; prec=tp/nE if nE else 0; f1=2*rec_*prec/(rec_+prec) if tp else 0
    res.append(dict(d=p[0],q=p[1],r=p[2],recall=round(rec_,3),precision=round(prec,3),f1=round(f1,3),n_detected=nE,
                    overlap_any_problem=round(anyprob/nE,3) if nE else None,clip_spearman=round(float(spearmanr(cnt,hum)[0]),3)))
best=max(res,key=lambda r:(r['f1'],r['d']))
json.dump({'grid':res,'selected':best,'n_human_uh':nH},open(LP.OUT/'disfl_dev.json','w'),indent=1,ensure_ascii=False)
for r in sorted(res,key=lambda r:-r['f1'])[:8]: print(r)
print('SELECTED',best)
