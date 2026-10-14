"""Independent recomputation of station-6 key numbers (numpy/scipy only, no sklearn)."""
import json, numpy as np
from scipy.stats import mannwhitneyu, spearmanr
import localpaths as LP
A=np.load(LP.OUT/'disfl_main_arrays.npz'); y=A['y']; X5=A['X5']; XD=A['XD']
rows=json.load(open(LP.OUT/'disfl_clip_rows.json',encoding='utf-8'))
gb=[r for r in rows if r['j'] in ('好','不好')]
bad=np.array([r['j']=='不好' for r in gb]); d3=np.array([r['d'][2] for r in gb])
u=mannwhitneyu(d3[bad],d3[~bad],alternative='greater'); auc=u.statistic/(bad.sum()*(~bad).sum())
out={'D3_auroc_mw':round(float(auc),3),'D3_mw_p':round(float(u.pvalue),4)}
def logit_fit(Z,y,C=1.0,it=100):
    Zi=np.c_[np.ones(len(Z)),Z]; w=np.zeros(Zi.shape[1]); P=np.eye(len(w))/C; P[0,0]=0
    for _ in range(it):
        p=1/(1+np.exp(-Zi@w)); g=Zi.T@(p-y)+P@w; H=Zi.T@(Zi*(p*(1-p))[:,None])+P
        step=np.linalg.solve(H,g); w-=step
        if np.abs(step).max()<1e-10: break
    return w
def auroc(y,s):
    r=np.argsort(np.argsort(s,kind='mergesort'))+1.0
    for v in np.unique(s): m=s==v; r[m]=r[m].mean()
    n1=y.sum(); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
def strat_folds(y,seed):
    # own stratified split (different from sklearn's) -> checks robustness, not identical folds
    rg=np.random.default_rng(seed); f=np.zeros(len(y),int)
    for c in (0,1):
        idx=np.where(y==c)[0]; rg.shuffle(idx); f[idx]=np.arange(len(idx))%5
    return f
def cv(X,reps=10):
    res=[]
    for r in range(reps):
        f=strat_folds(y,777+r); oof=np.zeros(len(y))
        for k in range(5):
            tr,te=f!=k,f==k; med=np.nanmedian(X[tr],0); a=np.where(np.isnan(X[tr]),med,X[tr]); b=np.where(np.isnan(X[te]),med,X[te])
            mu,sd=a.mean(0),a.std(0); w=logit_fit((a-mu)/sd,y[tr]); oof[te]=np.c_[np.ones(te.sum()),(b-mu)/sd]@w
        res.append(auroc(y,oof))
    return np.array(res)
a5=cv(X5); a5d=cv(np.c_[X5,XD])
out['own_cv_auroc_P5']=round(float(a5.mean()),3); out['own_cv_auroc_P5D']=round(float(a5d.mean()),3); out['own_delta']=round(float((a5d-a5).mean()),3)
# post-hoc descriptive: boundary-pause rate (phonation recomputed from the Praat segments)
import csv, disfl as D, disfl_io as IO
KEY={r['clip_id']:r for r in csv.DictReader(open(LP.LISTEN_KEY,encoding='utf-8'))}
bnd=np.array([r['det']['n_boundary']/(D.phonation(IO.rec(r['sid'])[1],(float(KEY[r['clip']]['start_s']),float(KEY[r['clip']]['end_s'])))/60) for r in gb])
ub=mannwhitneyu(bnd[bad],bnd[~bad]); out['posthoc_boundary_pause_rate_auroc_bad']=round(float(ub.statistic/(bad.sum()*(~bad).sum())),3); out['posthoc_n_ok']=len(bnd); out['posthoc_boundary_two_sided_p']=round(float(ub.pvalue),3)
out['posthoc_clip_D3_vs_pause_rate_spearman']=round(float(spearmanr(d3,X5[:,1],nan_policy='omit')[0]),3)
print(json.dumps(out,indent=1,ensure_ascii=False)); json.dump(out,open(LP.OUT/'disfl_verify.json','w'),indent=1)
