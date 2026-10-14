"""Whole-recording re-run of every listener-label analysis (user request 2026-09-29: use each recording in full).
Same rules as the frozen plans (stations 5, 6, 8); only the analysis window changes from the 30-s clip to the whole recording."""
import csv, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import features as F, disfl as D, disfl_io as IO, litcues as L
import localpaths as LP
U = str(LP.PRIV)
KEY = [r for r in csv.DictReader(open(f'{U}/listening_20260926/listening_key_DO_NOT_OPEN.csv', encoding='utf-8')) if r['repeat_of'] == '']
LAB = {r['clip_id']: r['judgement'] for r in csv.DictReader(open(LP.LISTEN_LABELS, encoding='utf-8-sig'))}
judg = {r['sample_id']: LAB[r['clip_id']] for r in KEY}
ids = sorted(judg)
rows = {}
for sid in ids:
    pr, seg, fr = IO.rec(sid); rec = json.load(open(LP.OUT/f'{sid}.json')); iv = (0.0, rec['duration_s'])
    syl = F.syllable_nuclei(pr["int_t"], pr["int_db"], pr["f0_t"], pr["f0"], seg)
    c = L.cue_row(pr, seg, fr, IO.words(sid), iv, syl)
    d = D.d_measures(seg, fr, IO.words(sid), iv, (0.4, 35, 3))
    p5 = [np.nan if rec['whole'][k] is None else rec['whole'][k] for k in F.P5]
    rows[sid] = dict(p5=p5, D1=d['D1'], D2=d['D2'], D3=d['D3'], cue=c)
gb = [s for s in ids if judg[s] in ('好', '不好')]; y = np.array([1 if judg[s] == '好' else 0 for s in gb])
rng = np.random.default_rng(20260929)
def perm_auc(v, yy):
    v = np.asarray(v, float); ok = ~np.isnan(v); v, yy = v[ok], yy[ok]; a = roc_auc_score(yy, v)
    pr = np.array([roc_auc_score(rng.permutation(yy), v) for _ in range(999)]); return round(float(a), 3), round(float((1 + np.sum(pr >= a)) / 1000), 3)
out = {'n': len(gb), 'n_good': int(y.sum())}
# expected direction: higher -> good
exp = {'articulation_rate': 1, 'pause_rate_per_min': -1, 'mean_pause_s': -1, 'f0_sd_st': 1, 'intensity_sd_db': 1}
out['P5_auroc_good'] = {k: round(float(roc_auc_score(y[~np.isnan(v)], (v * exp[k])[~np.isnan(v)])), 3) for i, k in enumerate(F.P5) for v in [np.array([rows[s]['p5'][i] for s in gb], float)]}
dd = {}
for k in ('D2', 'D3'):
    a, p = perm_auc([-rows[s][k] for s in gb], y); dd[k] = {'auroc_good': a, 'p_one_sided': p}
ps = sorted(dd, key=lambda k: dd[k]['p_one_sided'])
for i, k in enumerate(ps): dd[k]['p_holm'] = round(min(1, max(dd[q]['p_one_sided'] * (len(ps) - j) for j, q in enumerate(ps[:i + 1]))), 3)
dd['D1_descriptive'] = {'auroc_good': round(float(roc_auc_score(y, [-rows[s]['D1'] for s in gb])), 3)}
out['station6_univariate_whole'] = dd
# literature index (station 8 rules, standardised over the 82 whole recordings)
idx, X = L.index([rows[s]['cue'] for s in ids]); iw = {s: v for s, v in zip(ids, idx)}
a, p = perm_auc([iw[s] for s in gb], y); out['station8_index_whole'] = {'auroc_good': a, 'p_one_sided': p}
keys = list(L.SIGNS)
out['station8_components_whole'] = {k: round(float(roc_auc_score(y[~np.isnan(v)], v[~np.isnan(v)])), 3) for i, k in enumerate(keys) for v in [np.array([rows[s]['cue'][k] * L.SIGNS[k] for s in gb], float)]}
for k in ('L2', 'P1'):
    v = [rows[s]['cue'][k] * L.SIGNS[k] for s in gb]; a, p = perm_auc(v, y); out['station8_components_whole'][k + '_perm_p'] = p
# station 6 main test 2, whole recordings
X5 = np.array([rows[s]['p5'] for s in gb], float); XD = np.array([[rows[s]['D2'], rows[s]['D3']] for s in gb], float)
def impute(a, b):
    m = np.nanmedian(a, 0); return np.where(np.isnan(a), m, a), np.where(np.isnan(b), m, b)
def cv_auc(X):
    res = []
    for r in range(10):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=20260928 + r).split(X, y):
            a_, b_ = impute(X[tr], X[te]); sc = StandardScaler().fit(a_)
            oof[te] = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(a_), y[tr]).predict_proba(sc.transform(b_))[:, 1]
        res.append(roc_auc_score(y, oof))
    return np.array(res)
a5 = cv_auc(X5); a5d = cv_auc(np.c_[X5, XD]); delta = float((a5d - a5).mean())
perm = [float((cv_auc(np.c_[X5, XD[rng.permutation(len(y))]]) - a5).mean()) for _ in range(999)]
out['station6_main2_whole'] = {'auroc_P5': round(float(a5.mean()), 3), 'auroc_P5D': round(float(a5d.mean()), 3), 'delta': round(delta, 3),
                               'perm_p_one_sided': round(float((1 + np.sum(np.array(perm) >= delta)) / 1000), 3)}
# confidence (24 recordings) vs whole-recording measures
K = json.load(open(f'{U}/pause_check_20260928/pause_check_key_DO_NOT_OPEN.json', encoding='utf-8'))  # clip windows only
conf = {r['clip_id']: int(r['confidence']) for r in csv.DictReader(open(LP.PC_LABELS, encoding='utf-8-sig'))}
cs = [k['sample_id'] for k in K]; cv = [conf[k['clip_id']] for k in K]
cc = {k: round(float(spearmanr(cv, [rows[s]['p5'][i] for s in cs], nan_policy='omit')[0]), 3) for i, k in enumerate(F.P5)}
cc.update({k: round(float(spearmanr(cv, [rows[s][k] for s in cs])[0]), 3) for k in ('D1', 'D2', 'D3')})
r_, p_ = spearmanr(cv, [iw[s] for s in cs]); cc['literature_index'] = [round(float(r_), 3), round(float(p_), 3)]
out['confidence_vs_whole_recording'] = cc
json.dump(out, open(LP.OUT/'whole_rerun.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
