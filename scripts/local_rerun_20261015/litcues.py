"""Station 8: literature confidence cues (plan docs/research/2026-09-29-literature-cues-plan.md, sha 77885caf...)."""
import numpy as np
import features as F, disfl as D

def speaker_ref_db(pr, seg):
    m = F.in_segments(pr["int_t"], seg)
    return float(np.median(pr["int_db"][m]))

def sentence_ends(words, iv):
    a, b = iv
    return [(s, e) for s, e, t in words if s >= a and e <= b and t.strip()[-1:] in ".!"]

def L2_end_loudness(pr, seg, words, iv):
    ref = speaker_ref_db(pr, seg); vals = []
    for s, e in sentence_ends(words, iv):
        m = (pr["int_t"] >= s) & (pr["int_t"] <= e)
        if m.sum() >= 2: vals.append(float(pr["int_db"][m].mean()) - ref)
    return float(np.median(vals)) if len(vals) >= 2 else np.nan

def P1_final_rise(pr, words, iv, win=0.3, thr=1.0):
    f0t, f0 = pr["f0_t"], pr["f0"]; voiced_all = f0[f0 > 0]
    if len(voiced_all) < 20: return np.nan
    med = np.median(voiced_all); flags = []
    for s, e in sentence_ends(words, iv):
        m = (f0t >= max(s, e - win)) & (f0t <= e) & (f0 > 0)
        v = f0[m]
        if len(v) < 6: continue
        st = 12 * np.log2(v / med); k = len(st) // 3
        flags.append(st[-k:].mean() - st[:k].mean() > thr)
    return float(np.mean(flags)) if len(flags) >= 2 else np.nan

def cue_row(pr, seg, fr, words, iv, syl):
    p = F.features_in(iv, seg, pr, syl)
    d = D.d_measures(seg, fr, words, iv, (0.4, 35, 3))
    return {"AR": p["articulation_rate"], "F0R": p["f0_range_st"], "ISD": p["intensity_sd_db"], "D3": d["D3"],
            "PR": p["pause_rate_per_min"], "L2": L2_end_loudness(pr, seg, words, iv), "P1": P1_final_rise(pr, words, iv)}

SIGNS = {"AR": 1, "F0R": 1, "ISD": 1, "D3": -1, "L2": 1, "P1": -1}

def index(rows, ref=None, swap_pr=False):
    keys = list(SIGNS); X = np.array([[r[k] for k in keys] for r in rows], float)
    if swap_pr:
        X[:, keys.index("D3")] = [r["PR"] for r in rows]
    R = X if ref is None else ref
    mu, sd = np.nanmean(R, 0), np.nanstd(R, 0)
    Z = (X - mu) / sd; Z = np.where(np.isnan(Z), 0.0, Z)
    return (Z * np.array([SIGNS[k] for k in keys])).mean(1), X
