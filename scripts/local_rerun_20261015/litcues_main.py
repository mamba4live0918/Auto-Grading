import csv, json, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import features as F, disfl_io as IO, litcues as L
import localpaths as LP
U = str(LP.PRIV)
KEY = [r for r in csv.DictReader(open(f'{U}/listening_20260926/listening_key_DO_NOT_OPEN.csv', encoding='utf-8')) if r['repeat_of'] == '']
LAB = {r['clip_id']: r['judgement'] for r in csv.DictReader(open(LP.LISTEN_LABELS, encoding='utf-8-sig'))}
MAN = str(LP.MAN)
score = {r['sample_id']: float(r['total_score']) for r in csv.DictReader(open(MAN, encoding='utf-8')) if r['score_status'] == 'observed'}
syl_cache = {}
def row(sid, iv):
    pr, seg, fr = IO.rec(sid)
    if sid not in syl_cache: syl_cache[sid] = F.syllable_nuclei(pr["int_t"], pr["int_db"], pr["f0_t"], pr["f0"], seg)
    return L.cue_row(pr, seg, fr, IO.words(sid), iv, syl_cache[sid])
Lrows = [row(r['sample_id'], (float(r['start_s']), float(r['end_s']))) for r in KEY]
judg = [LAB[r['clip_id']] for r in KEY]
idx, XL = L.index(Lrows)
gb = [i for i, j in enumerate(judg) if j in ('好', '不好')]; y = np.array([1 if judg[i] == '好' else 0 for i in gb])
rng = np.random.default_rng(20260929)
def perm_auc(v, yy, n=999):
    ok = ~np.isnan(v); v, yy = v[ok], yy[ok]; a = roc_auc_score(yy, v)
    pr = np.array([roc_auc_score(rng.permutation(yy), v) for _ in range(n)]); return a, (1 + np.sum(pr >= a)) / (n + 1), int(ok.sum())
out = {'n_clips': len(Lrows), 'n_good_bad': int(len(gb)), 'n_good': int(y.sum())}
a, p, n = perm_auc(idx[gb], y); out['T1_index_auroc_good'] = {'auroc': round(a, 3), 'p_one_sided': round(p, 3), 'n': n}
keys = list(L.SIGNS); t2 = {}
for k in ('L2', 'P1'):
    v = XL[gb, keys.index(k)] * L.SIGNS[k]; a, p, n = perm_auc(v, y); t2[k] = {'auroc_in_literature_direction': round(a, 3), 'p_one_sided': round(p, 3), 'n': n}
ps = sorted(t2, key=lambda k: t2[k]['p_one_sided'])
for i, k in enumerate(ps): t2[k]['p_holm'] = round(min(1, max(t2[q]['p_one_sided'] * (len(ps) - j) for j, q in enumerate(ps[:i + 1]))), 3)
out['T2'] = t2
out['components_auroc_good_literature_direction'] = {k: round(roc_auc_score(y[~np.isnan(XL[gb, i])], (XL[gb, i] * L.SIGNS[k])[~np.isnan(XL[gb, i])]), 3) for i, k in enumerate(keys)}
out['missing_L_clips'] = {k: int(np.isnan(XL[:, i]).sum()) for i, k in enumerate(keys)}
idx_pr, _ = L.index(Lrows, swap_pr=True); out['control_index_with_all_pauses_instead_of_D3'] = round(roc_auc_score(y, idx_pr[gb]), 3)
# whole recordings vs teacher score
ids = sorted({r['sample_id'] for r in KEY})
Wrows = []
for sid in ids:
    pr, seg, fr = IO.rec(sid); dur = pr['f0_t'][-1]
    Wrows.append(row(sid, (0, float(max(pr['int_t'][-1], dur)))))
widx, _ = L.index(Wrows); sc = [score[s] for s in ids]
r_, p_ = spearmanr(widx, sc); out['whole_index_vs_total_score'] = {'spearman': round(float(r_), 3), 'p': round(float(p_), 3), 'n': len(ids)}
# P clips vs confidence
K = json.load(open(f'{U}/pause_check_20260928/pause_check_key_DO_NOT_OPEN.json', encoding='utf-8'))  # clip windows only
conf = {r['clip_id']: int(r['confidence']) for r in csv.DictReader(open(LP.PC_LABELS, encoding='utf-8-sig'))}
Prows = [row(k['sample_id'], (float(k['start_s']), float(k['start_s']) + 30)) for k in K]
pidx, XP = L.index(Prows, ref=XL); c = [conf[k['clip_id']] for k in K]
r_, p_ = spearmanr(pidx, c); out['P_index_vs_confidence'] = {'spearman': round(float(r_), 3), 'p': round(float(p_), 3), 'n': len(K)}
out['P_components_vs_confidence'] = {k: round(float(spearmanr(XP[:, i] * L.SIGNS[k], c, nan_policy='omit')[0]), 3) for i, k in enumerate(keys)}
r_, p_ = spearmanr(idx, [score[r['sample_id']] for r in KEY]); out['clip_index_vs_total_score'] = {'spearman': round(float(r_), 3), 'p': round(float(p_), 3)}
(LP.OUT/'litcues').mkdir(exist_ok=True)
json.dump(out, open(LP.OUT/'litcues'/'litcues_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
np.savez(LP.OUT/'litcues'/'litcues_arrays.npz', XL=XL, idx=idx, judg=np.array(judg), XP=XP, pidx=pidx, conf=np.array(c), widx=widx, sc=np.array(sc))
print(json.dumps(out, ensure_ascii=False, indent=1))
