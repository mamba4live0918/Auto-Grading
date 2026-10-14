"""Station 6 secondaries 2-3 on whole recordings (plan sha aaf0740d...). D = [D2, D3]; D1 descriptive only."""
import json, csv, numpy as np
from scipy.stats import spearmanr, t as tdist, fisher_exact
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
import features as F, model as M, disfl as D, disfl_io as IO
import localpaths as LP
MAN=str(LP.MAN)
rows=[r for r in csv.DictReader(open(MAN,encoding='utf-8')) if r['score_status']=='observed' and r['audio_status']=='succeeded' and r['feature_status']=='succeeded' and r['supervised_eligible']=='True']
ids=sorted(r['sample_id'] for r in rows); score={r['sample_id']:float(r['total_score']) for r in rows}
y=np.array([score[i] for i in ids]); n=len(ids); assert n==82
rec={i:json.load(open(LP.OUT/f'{i}.json')) for i in ids}
Xp=np.array([[np.nan if rec[i]['whole'][k] is None else rec[i]['whole'][k] for k in F.P5] for i in ids],float)
PFP=tuple(json.load(open(LP.OUT/'disfl_dev.json'))['selected'][k] for k in 'dqr')
Dw=[]
for i in ids:
    pr,seg,fr=IO.rec(i); dur=rec[i]['duration_s']
    d=D.d_measures(seg,fr,IO.words(i),(0,dur),PFP); Dw.append([d['D1'],d['D2'],d['D3']])
    assert abs(d['D2']-rec[i]['whole']['long_pause_rate_per_min'])<1e-6, (i,d['D2'],rec[i]['whole']['long_pause_rate_per_min'])
Dw=np.array(Dw); XD=Dw[:,1:]
out={'n':n,'D_whole_median':{k:float(np.median(Dw[:,j])) for j,k in enumerate(['D1','D2','D3'])}}
out['D_vs_score_spearman_descriptive']={k:[round(float(v),3) for v in spearmanr(Dw[:,j],y)] for j,k in enumerate(['D1','D2','D3'])}
# ---- secondary 2: Test A redo ----
splits=M.outer_splits(n)
conds={'B0_median':('median',None),'Ridge_P5':('ridge',Xp),'Ridge_P5D':('ridge',np.c_[Xp,XD])}
preds,alph=M.run(conds,y,splits)
out['models']=[{'model':k,'mae':float(M.mae(p,y).mean()),'oof_spearman':float(np.nanmean([spearmanr(p[r],y)[0] if np.std(p[r])>0 else np.nan for r in range(M.R)]))} for k,p in preds.items()]
rng=np.random.default_rng(20260926);draws=rng.integers(0,n,(5000,n))
def contrast(a,b):
    d=(np.abs(preds[a]-y)-np.abs(preds[b]-y)); per=d.mean(0);lo,hi=np.quantile(per[draws].mean(1),[.025,.975])
    dj=np.array([d[r,te].mean() for r,folds in enumerate(splits) for tr,te in folds])
    J=len(dj);ratio=(n/M.K)/(n-n/M.K);se=np.sqrt((1/J+ratio)*dj.var(ddof=1));tt=dj.mean()/se
    return {'candidate':a,'baseline':b,'mae_diff':round(float(per.mean()),3),'boot95':[round(float(lo),3),round(float(hi),3)],'nb_p_one_sided':round(float(tdist.cdf(tt,J-1)),3)}
out['contrasts']=[contrast('Ridge_P5D','B0_median'),contrast('Ridge_P5D','Ridge_P5')]
# ---- secondary 3: grouping redo with P5+D, then listener labels ----
Xs=np.c_[Xp,XD]; Z=np.where(np.isnan(Xs),np.nanmedian(Xs,0),Xs); Z=(Z-Z.mean(0))/Z.std(0)
L=linkage(Z,'ward'); sil={k:float(silhouette_score(Z,fcluster(L,k,'maxclust'))) for k in range(2,6)}
best=max(sil.values()); k=min(kk for kk,v in sil.items() if v>=best-0.02); g=fcluster(L,k,'maxclust')
rg=np.random.default_rng(20260926); aris=[]
for b in range(200):
    idx=np.sort(rg.choice(n,int(0.8*n),replace=False)); aris.append(adjusted_rand_score(g[idx],fcluster(linkage(Z[idx],'ward'),k,'maxclust')))
KEY=[r for r in csv.DictReader(open(LP.LISTEN_KEY,encoding='utf-8')) if r['repeat_of']=='']
LAB={r['clip_id']:r['judgement'] for r in csv.DictReader(open(LP.LISTEN_LABELS,encoding='utf-8-sig'))}
per={r['sample_id']:LAB[r['clip_id']] for r in KEY}
tab={int(c):{'好':0,'不好':0,'说不准':0} for c in set(g)}
for i,c in zip(ids,g): tab[int(c)][per[i]]+=1
grp={int(c):{'n':int((g==c).sum()),**{k2:round(float(np.nanmean(Xs[g==c,j])),2) for j,k2 in enumerate(F.P5+['D2','D3'])},'score_mean':round(float(y[g==c].mean()),1),'labels':tab[int(c)]} for c in set(g)}
out['grouping']={'silhouette':{k2:round(v,3) for k2,v in sil.items()},'k':k,'ari_median':round(float(np.median(aris)),3),'ari_p10':round(float(np.percentile(aris,10)),3),'groups':grp}
if k==2:
    t=[[tab[c]['好'],tab[c]['不好']] for c in sorted(tab)]; out['grouping']['fisher_p']=round(float(fisher_exact(t)[1]),3)
# whole-recording univariate vs listener good/bad (descriptive)
gbi=[j for j,i in enumerate(ids) if per[i] in ('好','不好')]; yb=np.array([1 if per[ids[j]]=='不好' else 0 for j in gbi])
out['whole_auroc_predicting_bad_descriptive']={k2:round(float(roc_auc_score(yb,Dw[gbi,j])),3) for j,k2 in enumerate(['D1','D2','D3'])}
json.dump(out,open(LP.OUT/'disfl_whole.json','w',encoding='utf-8'),indent=1,ensure_ascii=False)
print(json.dumps(out,indent=1,ensure_ascii=False))
