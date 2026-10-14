"""Independent recomputation (numpy only) of the whole-recording main test 2 and D3 AUROC."""
import csv, json, numpy as np
import features as F, disfl as D, disfl_io as IO
import localpaths as LP
U=str(LP.PRIV)
KEY=[r for r in csv.DictReader(open(f'{U}/listening_20260926/listening_key_DO_NOT_OPEN.csv',encoding='utf-8')) if r['repeat_of']=='']
LAB={r['clip_id']:r['judgement'] for r in csv.DictReader(open(LP.LISTEN_LABELS,encoding='utf-8-sig'))}
j={r['sample_id']:LAB[r['clip_id']] for r in KEY}; gb=sorted(s for s in j if j[s] in('好','不好'))
y=np.array([1 if j[s]=='好' else 0 for s in gb])
X5=[];XD=[]
for s in gb:
    rec=json.load(open(LP.OUT/f'{s}.json')); pr,seg,fr=IO.rec(s)
    d=D.d_measures(seg,fr,IO.words(s),(0,rec['duration_s']),(0.4,35,3))
    X5.append([np.nan if rec['whole'][k] is None else rec['whole'][k] for k in F.P5]); XD.append([d['D2'],d['D3']])
X5=np.array(X5,float); XD=np.array(XD,float)
def auroc(y,s):
    r=np.argsort(np.argsort(s,kind='mergesort'))+1.0
    for v in np.unique(s): m=s==v; r[m]=r[m].mean()
    n1=y.sum(); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
def fit(Z,y,C=1.0):
    Zi=np.c_[np.ones(len(Z)),Z]; w=np.zeros(Zi.shape[1]); P=np.eye(len(w))/C; P[0,0]=0
    for _ in range(100):
        p=1/(1+np.exp(-Zi@w)); g=Zi.T@(p-y)+P@w; H=Zi.T@(Zi*(p*(1-p))[:,None])+P; st=np.linalg.solve(H,g); w-=st
        if np.abs(st).max()<1e-10: break
    return w
def folds(seed):
    rg=np.random.default_rng(seed); f=np.zeros(len(y),int)
    for c in (0,1):
        i=np.where(y==c)[0]; rg.shuffle(i); f[i]=np.arange(len(i))%5
    return f
def cv(X):
    out=[]
    for r in range(10):
        f=folds(555+r); oof=np.zeros(len(y))
        for k in range(5):
            tr,te=f!=k,f==k; m=np.nanmedian(X[tr],0); a=np.where(np.isnan(X[tr]),m,X[tr]); b=np.where(np.isnan(X[te]),m,X[te])
            mu,sd=a.mean(0),a.std(0); w=fit((a-mu)/sd,y[tr]); oof[te]=np.c_[np.ones(te.sum()),(b-mu)/sd]@w
        out.append(auroc(y,oof))
    return np.mean(out)
a5=cv(X5); a5d=cv(np.c_[X5,XD])
print(json.dumps({'own_auroc_P5':round(a5,3),'own_auroc_P5D':round(a5d,3),'own_delta':round(a5d-a5,3),'D3_auroc_good':round(auroc(y,-XD[:,1]),3)}))
