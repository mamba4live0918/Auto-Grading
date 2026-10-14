"""Repeated 10x5 CV, nested ridge, permutation test (plan 2026-09-26)."""
import numpy as np
from sklearn.model_selection import KFold
ALPHAS=np.logspace(-2,4,25)
R,K=10,5
def outer_splits(n):
    return [[(tr,te) for tr,te in KFold(K,shuffle=True,random_state=20260926+r).split(np.zeros(n))] for r in range(R)]
def prep(Xtr,Xte):
    med=np.nanmedian(Xtr,axis=0)
    Xtr=np.where(np.isnan(Xtr),med,Xtr);Xte=np.where(np.isnan(Xte),med,Xte)
    mu=Xtr.mean(0);sd=Xtr.std(0);sd[sd<1e-12]=1.0
    return (Xtr-mu)/sd,(Xte-mu)/sd
def ridge_path(Z,y,Zte,alphas):
    """predictions for all alphas: shape (len(alphas), n_test)"""
    ym=y.mean();U,s,Vt=np.linalg.svd(Z,full_matrices=False)
    Uy=U.T@(y-ym)
    d=s[None,:]/(s[None,:]**2+alphas[:,None])          # (A,p)
    coef=(d*Uy[None,:])@Vt                              # (A,p)
    return ym+Zte@coef.T                                # (n_test,A)
def fit_ridge_nested(X,y,tr,te,seed):
    inner=KFold(5,shuffle=True,random_state=seed).split(tr)
    err=np.zeros(len(ALPHAS));cnt=0
    for a,b in inner:
        Za,Zb=prep(X[tr[a]],X[tr[b]])
        P=np.clip(ridge_path(Za,y[tr[a]],Zb,ALPHAS),0,100)
        err+=np.abs(P-y[tr[b]][:,None]).sum(0);cnt+=len(b)
    k=int(np.argmin(err));Ztr,Zte=prep(X[tr],X[te])
    return np.clip(ridge_path(Ztr,y[tr],Zte,ALPHAS[k:k+1])[:,0],0,100),ALPHAS[k]
def fit_ols(X,y,tr,te):
    Ztr,Zte=prep(X[tr],X[te]);A=np.c_[np.ones(len(tr)),Ztr]
    w=np.linalg.lstsq(A,y[tr],rcond=None)[0];return np.clip(np.c_[np.ones(len(te)),Zte]@w,0,100)
def run(conditions,y,splits):
    """conditions: dict name->('median'|'mean'|'ols'|'ridge', X). returns preds[name] (R,n), alphas"""
    n=len(y);preds={k:np.full((R,n),np.nan) for k in conditions};alph={k:[] for k in conditions}
    for r,folds in enumerate(splits):
        for f,(tr,te) in enumerate(folds):
            for name,(kind,X) in conditions.items():
                if kind=='median': preds[name][r,te]=np.median(y[tr])
                elif kind=='mean': preds[name][r,te]=np.mean(y[tr])
                elif kind=='ols': preds[name][r,te]=fit_ols(X,y,tr,te)
                else:
                    p,a=fit_ridge_nested(X,y,tr,te,seed=1000*r+f);preds[name][r,te]=p;alph[name].append(a)
    return preds,alph
def mae(pred,y): return np.abs(pred-y[None,:]).mean(1)          # per repeat
def rmse(pred,y): return np.sqrt(((pred-y[None,:])**2).mean(1))
