import json,sys,numpy as np
from concurrent.futures import ProcessPoolExecutor
import model as M
import localpaths as LP
d=np.load(LP.OUT/'preds.npz');y=d['y'];n=len(y)
import analysis_inputs as AI
Xp=AI.Xp; splits=M.outer_splits(n)
def delta(yy):
    p,_=M.run({'B0':('median',None),'R':('ridge',Xp)},yy,splits)
    return float(M.mae(p['R'],yy).mean()-M.mae(p['B0'],yy).mean())
def chunk(seed):
    rng=np.random.default_rng(seed);return [delta(rng.permutation(y)) for _ in range(111)]
if __name__=='__main__':
    obs=delta(y)
    with ProcessPoolExecutor(9) as ex: perm=[v for c in ex.map(chunk,range(9)) for v in c][:999]  # 9 workers (cloud: 2); same seeds, same result
    perm=np.array(perm);p=(1+np.sum(perm<=obs))/(1+len(perm))
    res={'observed_delta':obs,'n_perm':len(perm),'p_one_sided':float(p),'perm_quantiles':np.quantile(perm,[.01,.05,.5,.95]).tolist()}
    np.save(LP.OUT/'perm.npy',perm);json.dump(res,open(LP.OUT/'permutation.json','w'),indent=1);print(res)
