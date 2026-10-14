import json,csv,re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist
import features as F, model as M
import localpaths as LP
MAN=str(LP.MAN)
TR=str(LP.TR_CHECK)
rows=[r for r in csv.DictReader(open(MAN,encoding='utf-8')) if r['score_status']=='observed' and r['audio_status']=='succeeded' and r['feature_status']=='succeeded' and r['supervised_eligible']=='True']
ids=sorted(r['sample_id'] for r in rows); score={r['sample_id']:float(r['total_score']) for r in rows}
y=np.array([score[i] for i in ids]); n=len(ids); assert n==82
rec={i:json.load(open(LP.OUT/f'{i}.json')) for i in ids}
def mat(part,names): return np.array([[np.nan if rec[i][part][k] is None else rec[i][part][k] for k in names] for i in ids],float)
Xp=mat('whole',F.P5); Xe=mat('whole',F.EXT)
eg=pd.concat([pd.read_csv(LP.EGEMAPS_DIR/f'{i}.csv',sep=';') for i in ids],ignore_index=True)
assert list(eg['name'].str.strip("'"))==ids
Xg=eg.drop(columns=['name','frameTime']).to_numpy(float)
OLD11=["speech_ratio","speech_segment_duration_median_seconds","internal_pauses_per_speech_minute","internal_pause_duration_median_seconds","acoustic_envelope_peaks_per_speech_minute","envelope_peak_interval_cv","pitch_p10_p90_range_semitones","pitch_std_semitones","rms_robust_cv","spectral_flux_dispersion","candidate_transients_per_speech_minute"]
out={'n':n,'score':{'min':y.min(),'max':y.max(),'median':float(np.median(y)),'sd':float(y.std(ddof=1))}}
# ---------------- signal checks (descriptive) ----------------
def rho(a,b):
    m=np.isfinite(a)&np.isfinite(b); return float(spearmanr(a[m],b[m])[0]),int(m.sum())
rel={}
for k in F.EXT:
    a=mat('half1',[k])[:,0];b=mat('half2',[k])[:,0];w=mat('whole',[k])[:,0];f=mat('first48',[k])[:,0]
    r,m=rho(a,b); rel[k]={'split_half_rho':r,'spearman_brown':2*r/(1+r) if r>-1 else None,'n':m,'first48_vs_whole_rho':rho(f,w)[0]}
out['reliability']=rel
out['detector_agreement_pause_rate_rho']=rho(Xp[:,1],np.array([rec[i]['vad_pause_rate_per_min'] for i in ids]))[0]
out['speech_time_praat_vs_vad_rho']=rho(np.array([rec[i]['praat_speech_s'] for i in ids]),np.array([rec[i]['vad_speech_s'] for i in ids]))[0]
# ASR check: orthographic syllables per phonation second vs articulation rate
def osyl(w):
    w=re.sub(r'[^a-z]','',w.lower())
    if not w: return 0
    c=len(re.findall(r'[aeiouy]+',w)); c-= 1 if w.endswith('e') and not w.endswith(('le','ee')) and c>1 else 0
    return max(c,1)
asr={}
for line in open(TR,encoding='utf-8'):
    d=json.loads(line)
    if d['sample_id'] in score and d['status']=='succeeded':
        ws=[w['word'] for s in d['transcription']['segments'] for w in s.get('words',[])]
        asr[d['sample_id']]=(len(ws),sum(osyl(w) for w in ws))
asr_syl=np.array([asr[i][1] if i in asr else np.nan for i in ids]); nsyl=mat('whole',['n_syllables'])[:,0]; phon=mat('whole',['phonation_s'])[:,0]
out['asr_check']={'rho_syllable_counts':rho(nsyl,asr_syl)[0],'rho_rates':rho(nsyl/phon,asr_syl/phon)[0],
   'median_ratio_praat_over_asr':float(np.nanmedian(nsyl/asr_syl)),'n':int(np.isfinite(asr_syl).sum())}
C=np.array([[spearmanr(Xp[:,j],Xp[:,k],nan_policy='omit')[0] for k in range(5)] for j in range(5)])
out['p5_intercorrelation']=C.round(3).tolist()
uni=[]
for j,k in enumerate(F.P5):
    m=np.isfinite(Xp[:,j]); r,p=spearmanr(Xp[m,j],y[m]); uni.append({'feature':k,'rho':float(r),'p':float(p),'n':int(m.sum())})
ps=np.array([u['p'] for u in uni]);order=np.argsort(ps);adj=np.empty(5);run=0
for rank,j in enumerate(order): run=max(run,min(1,(5-rank)*ps[j])); adj[j]=run
for u,a in zip(uni,adj): u['p_holm']=float(a)
out['univariate']=uni
out['missing']={k:int(np.isnan(Xe[:,j]).sum()) for j,k in enumerate(F.EXT)}
out['descriptives']={k:{'median':float(np.nanmedian(Xe[:,j])),'p10':float(np.nanpercentile(Xe[:,j],10)),'p90':float(np.nanpercentile(Xe[:,j],90))} for j,k in enumerate(F.EXT)}
# ---------------- models ----------------
splits=M.outer_splits(n)
conds={'B0_median':('median',None),'B1_mean':('mean',None),'OLS_P5':('ols',Xp),'Ridge_P5':('ridge',Xp),
       'Ridge_E':('ridge',Xe),'Ridge_G':('ridge',Xg)}
preds,alph=M.run(conds,y,splits)
res=[]
for k,p in preds.items():
    sp=[spearmanr(p[r],y)[0] if np.std(p[r])>0 else np.nan for r in range(M.R)]
    res.append({'model':k,'mae':float(M.mae(p,y).mean()),'mae_sd_repeats':float(M.mae(p,y).std(ddof=1)),'rmse':float(M.rmse(p,y).mean()),
                'oof_spearman':float(np.nanmean(sp)),'alpha_median':float(np.median(alph[k])) if alph[k] else None,
                'alpha_at_max':int(np.sum(np.isclose(alph[k],M.ALPHAS[-1]))) if alph[k] else None})
out['models']=res
# contrasts vs B0: bootstrap by recording + Nadeau-Bengio
rng=np.random.default_rng(20260926);draws=rng.integers(0,n,(5000,n))
def contrast(a,b):
    d=(np.abs(preds[a]-y)-np.abs(preds[b]-y))              # (R,n)
    per=d.mean(0);lo,hi=np.quantile(per[draws].mean(1),[.025,.975])
    dj=np.array([d[r,te].mean() for r,folds in enumerate(splits) for tr,te in folds])
    J=len(dj);ratio=(n/M.K)/(n-n/M.K);se=np.sqrt((1/J+ratio)*dj.var(ddof=1));tt=dj.mean()/se
    return {'candidate':a,'baseline':b,'mae_diff':float(per.mean()),'boot95':[float(lo),float(hi)],'nb_t':float(tt),'nb_p_one_sided':float(tdist.cdf(tt,J-1))}
out['contrasts']=[contrast(k,'B0_median') for k in ['Ridge_P5','OLS_P5','Ridge_E','Ridge_G','B1_mean']]
np.savez(LP.OUT/'preds.npz',ids=np.array(ids),y=y,**preds)
json.dump(out,open(LP.OUT/'analysis.json','w'),indent=1,default=float)
print(json.dumps({k:out[k] for k in ['models','contrasts']},indent=1,default=float))
