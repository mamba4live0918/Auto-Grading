"""Independent recomputation with scikit-learn estimators + leakage check."""
import numpy as np, json
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import analysis_inputs as AI, model as M
import localpaths as LP
d=np.load(LP.OUT/'preds.npz'); y=d['y']; X=AI.Xp; n=len(y)
def sk_nested(tr,te,seed,yy):
    best=None
    for a in M.ALPHAS:
        e=0
        for ia,ib in KFold(5,shuffle=True,random_state=seed).split(tr):
            p=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=a)).fit(X[tr[ia]],yy[tr[ia]])
            e+=np.abs(np.clip(p.predict(X[tr[ib]]),0,100)-yy[tr[ib]]).sum()
        if best is None or e<best[0]-1e-9: best=(e,a)
    p=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=best[1])).fit(X[tr],yy[tr])
    return np.clip(p.predict(X[te]),0,100),best[1]
pred=np.full((M.R,n),np.nan)
for r,folds in enumerate(M.outer_splits(n)):
    for f,(tr,te) in enumerate(folds): pred[r,te]=sk_nested(tr,te,1000*r+f,y)[0]
maxdiff=float(np.abs(pred-d['Ridge_P5']).max())
b0=M.mae(d['B0_median'],y).mean(); delta=M.mae(pred,y).mean()-b0
# leakage: scramble held-out labels of one fold -> same alpha and predictions
tr,te=M.outer_splits(n)[3][2]; y2=y.copy(); y2[te]=np.random.default_rng(1).permutation(y2[te])+50
p1,a1=M.fit_ridge_nested(X,y,tr,te,seed=3002); p2,a2=M.fit_ridge_nested(X,y2,tr,te,seed=3002)
res={'sklearn_vs_closed_form_max_pred_diff':maxdiff,'sklearn_delta':float(delta),'leakage_same_alpha':bool(a1==a2),'leakage_max_pred_diff':float(np.abs(p1-p2).max())}
json.dump(res,open(LP.OUT/'verify.json','w'),indent=1);print(res)
