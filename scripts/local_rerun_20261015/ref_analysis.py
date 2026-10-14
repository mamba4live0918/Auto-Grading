import sys,json,glob,numpy as np
import localpaths as LP
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
import features as F
import analysis_inputs as AI
names={'articulation_rate':'发音速率','pause_rate_per_min':'停顿频率','mean_pause_s':'平均停顿时长','f0_sd_st':'音高波动','intensity_sd_db':'音量波动'}
man=json.load(open(LP.REF_MANIFEST,encoding='utf-8'))
gender={c['speaker_hash']:None for c in man['clips']}
prof=[];bad=[]
for f in sorted(glob.glob(str(LP.REF_OUT/'*.json'))):
    d=json.load(open(f))
    if 'error' in d or d.get('whole') is None: bad.append(Path(f).stem); continue
    prof.append([np.nan if d['whole'][k] is None else d['whole'][k] for k in F.P5])
Xp=np.array(prof,float); Xs=AI.Xp; y_score=np.load(LP.OUT/'preds.npz')['y']
out={'n_prof':len(Xp),'n_prof_failed':len(bad),'n_students':len(Xs)}
# descriptives
out['describe']={k:{'prof_median':float(np.nanmedian(Xp[:,j])),'prof_p10':float(np.nanpercentile(Xp[:,j],10)),'prof_p90':float(np.nanpercentile(Xp[:,j],90)),
   'stud_median':float(np.nanmedian(Xs[:,j])),'stud_p10':float(np.nanpercentile(Xs[:,j],10)),'stud_p90':float(np.nanpercentile(Xs[:,j],90)),
   'students_outside_prof_p10_p90':int(np.sum((Xs[:,j]<np.nanpercentile(Xp[:,j],10))|(Xs[:,j]>np.nanpercentile(Xp[:,j],90))))} for j,k in enumerate(F.P5)}
# R1 per-feature AUROC (professional=1)
rng=np.random.default_rng(20260926)
r1=[]
for j,k in enumerate(F.P5):
    a=Xp[:,j][np.isfinite(Xp[:,j])]; b=Xs[:,j][np.isfinite(Xs[:,j])]
    def auc(a,b): return roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b])
    bs=[auc(rng.choice(a,len(a)),rng.choice(b,len(b))) for _ in range(2000)]
    r1.append({'feature':k,'auroc':float(auc(a,b)),'ci95':np.quantile(bs,[.025,.975]).tolist()})
out['R1_single']=r1
# R1 multivariate + R2 OOF professional-likeness for students
X=np.r_[Xp,Xs]; lab=np.r_[np.ones(len(Xp)),np.zeros(len(Xs))].astype(int)
aucs=[];oof=np.zeros((10,len(X)))
for r in range(10):
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=20260926+r).split(X,lab):
        pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000))
        g=GridSearchCV(pipe,{'logisticregression__C':np.logspace(-3,3,13)},scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=r)).fit(X[tr],lab[tr])
        oof[r,te]=g.predict_proba(X[te])[:,1]
    aucs.append(roc_auc_score(lab,oof[r]))
out['R1_multi']={'auroc_mean':float(np.mean(aucs)),'auroc_min':float(np.min(aucs)),'auroc_max':float(np.max(aucs))}
full=GridSearchCV(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000)),{'logisticregression__C':np.logspace(-3,3,13)},scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=0)).fit(X,lab)
coef=full.best_estimator_[-1].coef_[0]; out['R1_multi']['full_data_standardized_coef']={k:float(c) for k,c in zip(F.P5,coef)}
plike=oof[:,len(Xp):].mean(0)
out['R2']={'plike_median':float(np.median(plike)),'plike_p10':float(np.percentile(plike,10)),'plike_p90':float(np.percentile(plike,90)),
   'spearman_with_total_score':float(spearmanr(plike,y_score)[0]),'p':float(spearmanr(plike,y_score)[1])}
# R3 grouping
Z=np.where(np.isnan(Xs),np.nanmedian(Xs,0),Xs); Z=(Z-Z.mean(0))/Z.std(0)
L=linkage(Z,'ward'); sil={k:float(silhouette_score(Z,fcluster(L,k,'maxclust'))) for k in range(2,6)}
best=max(sil.values()); k=min(kk for kk,v in sil.items() if v>=best-0.02)
g=fcluster(L,k,'maxclust')
aris=[]
for b in range(200):
    idx=np.sort(rng.choice(len(Z),int(.8*len(Z)),replace=False))
    gb=fcluster(linkage(Z[idx],'ward'),k,'maxclust'); aris.append(adjusted_rand_score(g[idx],gb))
groups=[]
for c in sorted(set(g)):
    m=g==c
    groups.append({'group':int(c),'n':int(m.sum()),'feature_medians':{kk:float(np.nanmedian(Xs[m,j])) for j,kk in enumerate(F.P5)},
       'plike_mean':float(plike[m].mean()),'total_score_mean':float(y_score[m].mean())})
out['R3']={'silhouette':sil,'k':k,'ari_median':float(np.median(aris)),'ari_p10':float(np.percentile(aris,10)),'groups':groups}
np.savez(LP.RUN/'students_ref.npz',ids=np.array(AI.ids),plike=plike,group=g)
json.dump(out,open(LP.OUT/'ref_results.json','w',encoding='utf-8'),indent=1,ensure_ascii=False)
print(json.dumps(out,indent=1,ensure_ascii=False))
