"""Station 6 main + secondary analyses on held-out L clips and whole recordings (plan sha aaf0740d...).
D1 failed the development criterion, so D = [D2, D3] in all confirmatory models."""
import json, csv, numpy as np
import features as F, disfl as D, disfl_io as IO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import localpaths as LP
DEV = json.load(open(LP.OUT/'disfl_dev.json'))['selected']; PFP = (DEV['d'], DEV['q'], DEV['r'])
KEY = [r for r in csv.DictReader(open(LP.LISTEN_KEY, encoding='utf-8')) if r['repeat_of'] == '']
LAB = {r['clip_id']: r for r in csv.DictReader(open(LP.LISTEN_LABELS, encoding='utf-8-sig'))}
def feats(sid, iv):
    pr, seg, fr = IO.rec(sid)
    syl = F.syllable_nuclei(pr["int_t"], pr["int_db"], pr["f0_t"], pr["f0"], seg)
    p = F.features_in(iv, seg, pr, syl)
    d = D.d_measures(seg, fr, IO.words(sid), iv, PFP)
    return [p[k] for k in F.P5], [d['D1'], d['D2'], d['D3']], d
rows = []
for r in KEY:
    s, e = float(r['start_s']), float(r['end_s']); p5, dd, det = feats(r['sample_id'], (s, e))
    note = LAB[r['clip_id']]['note']; j = LAB[r['clip_id']]['judgement']
    rows.append(dict(clip=r['clip_id'], sid=r['sample_id'], j=j, disfl_note=int(any(k in note.lower() for k in ('uh', '卡顿', '重复'))), p5=p5, d=dd, det=det))
out = {'n_clips': len(rows), 'fp_params_frozen': PFP}
# ---------- main test 2 ----------
gb = [x for x in rows if x['j'] in ('好', '不好')]
y = np.array([1 if x['j'] == '好' else 0 for x in gb])
X5 = np.array([x['p5'] for x in gb], float); XD = np.array([[x['d'][1], x['d'][2]] for x in gb], float)
def impute(tr, te):
    med = np.nanmedian(tr, 0); tr = np.where(np.isnan(tr), med, tr); te = np.where(np.isnan(te), med, te); return tr, te
def cv_auc(X, y, reps=10):
    aucs = []
    for r in range(reps):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=20260928 + r).split(X, y):
            a, b = impute(X[tr], X[te]); sc = StandardScaler().fit(a)
            m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(a), y[tr]); oof[te] = m.predict_proba(sc.transform(b))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return np.array(aucs)
a5 = cv_auc(X5, y); a5d = cv_auc(np.c_[X5, XD], y); delta = a5d - a5
rng = np.random.default_rng(20260928); perm = []
for i in range(999):
    pi = rng.permutation(len(y)); perm.append((cv_auc(np.c_[X5, XD[pi]], y) - a5).mean())
perm = np.array(perm)
out['main2'] = {'n': int(len(y)), 'n_good': int(y.sum()), 'auroc_P5_mean': round(float(a5.mean()), 3), 'auroc_P5_range': [round(float(a5.min()), 3), round(float(a5.max()), 3)],
                'auroc_P5D_mean': round(float(a5d.mean()), 3), 'auroc_P5D_range': [round(float(a5d.min()), 3), round(float(a5d.max()), 3)],
                'delta_mean': round(float(delta.mean()), 3), 'delta_range': [round(float(delta.min()), 3), round(float(delta.max()), 3)],
                'perm_p_one_sided': round(float((1 + np.sum(perm >= delta.mean())) / 1000), 3)}
np.savez(LP.OUT/'disfl_main_arrays.npz', y=y, X5=X5, XD=XD, a5=a5, a5d=a5d, perm=perm)
# ---------- secondary 1: univariate AUROC for predicting 不好 ----------
def perm_auc(v, yb, seed=20260928):
    ok = ~np.isnan(v); v, yb = v[ok], yb[ok]; a = roc_auc_score(yb, v); rg = np.random.default_rng(seed)
    pr = np.array([roc_auc_score(rg.permutation(yb), v) for _ in range(999)]); return a, (1 + np.sum(pr >= a)) / 1000, int(ok.sum())
bad = 1 - y; uni = {}
for k, col in (('D2_long_pause', XD[:, 0]), ('D3_mid_clause_pause', XD[:, 1])):
    a, p, n = perm_auc(col, bad); uni[k] = {'auroc_predicting_bad': round(float(a), 3), 'p_one_sided': round(float(p), 3), 'n': n}
ps = sorted(uni, key=lambda k: uni[k]['p_one_sided'])
for i, k in enumerate(ps): uni[k]['p_holm'] = round(min(1, max(uni[q]['p_one_sided'] * (len(ps) - j) for j, q in enumerate(ps[:i + 1]))), 3)
d1 = np.array([x['d'][0] for x in gb]); uni['D1_filled_pause_descriptive_only'] = {'auroc_predicting_bad': round(float(roc_auc_score(bad, d1)), 3)}
out['secondary1_clip'] = uni
dn = np.array([x['disfl_note'] for x in rows]); out['D1_vs_note_descriptive_only'] = {'auroc': round(float(roc_auc_score(dn, [x['d'][0] for x in rows])), 3), 'n_note': int(dn.sum()), 'n': len(rows)}
out['D2_D3_vs_note_descriptive'] = {k: round(float(roc_auc_score(dn, [x['d'][i] for x in rows])), 3) for k, i in (('D2', 1), ('D3', 2))}
json.dump(out, open(LP.OUT/'disfl_main.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump([{k: v for k, v in x.items() if k != 'det'} | {'det': x['det']} for x in rows], open(LP.OUT/'disfl_clip_rows.json', 'w', encoding='utf-8'))
print(json.dumps(out, indent=1, ensure_ascii=False))
