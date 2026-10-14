"""Independent recomputation of the Station 7 key numbers (plan section "核对").

numpy / scipy only, no scikit-learn, and its own counting of fillers and repetitions (union-find instead of
the interval merge in verbatim.py), its own AUROC, its own logistic regression and its own cross-validation splits.
The permutation tests use the same seeds as verbatim_main.py, so the p-values should reproduce; the cross-validated
AUROCs use different folds, so they should agree only approximately.

Run: .venv311\\Scripts\\python.exe scripts\\acoustic_redo\\verbatim_verify.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import localpaths as LP
ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "data" / "private"
FILL = {"uh", "um", "uhm", "umm", "er", "erm", "ah", "eh", "hmm", "mm"}


def strip_edges(word):
    a, b = 0, len(word)
    while a < b and not word[a].isalnum():
        a += 1
    while b > a and not word[b - 1].isalnum():
        b -= 1
    return word[a:b]


def own_counts(words, window=None):
    """(number of filler words, number of repetition sites) among words wholly inside `window`."""
    toks = [(s, e, strip_edges(w.lower())) for s, e, w in words if window is None or (s >= window[0] and e <= window[1])]
    toks = [t for t in toks if t[2]]
    fillers = sum(1 for t in toks if t[2] in FILL)
    kept = [t[2] for t in toks if t[2] not in FILL]
    parent = list(range(len(kept)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    hit = np.zeros(len(kept), bool)
    for i in range(len(kept) - 1):
        if kept[i] == kept[i + 1]:
            for k in (i, i + 1):
                hit[k] = True
            union(i, i + 1)
        if i + 3 < len(kept) and kept[i:i + 2] == kept[i + 2:i + 4]:
            for k in range(i, i + 4):
                hit[k] = True
                union(i, k)
    sites = {find(i) for i in range(len(kept)) if hit[i]}
    return fillers, len(sites)


def own_auc(y, s):
    """AUROC of score s for label y == 1 (average ranks for ties)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    n1 = int(y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def own_perm_p(v, bad, seed=20260929, n=999):
    ok = ~np.isnan(v)
    v, b = v[ok], bad[ok].astype(int)
    obs = own_auc(b, v)
    rng = np.random.default_rng(seed)
    count = sum(own_auc(rng.permutation(b), v) >= obs - 1e-12 for _ in range(n))
    return obs, (1 + count) / (n + 1)


def logit_fit(Z, y, C=1.0, iters=100):
    Zi = np.c_[np.ones(len(Z)), Z]
    w = np.zeros(Zi.shape[1])
    P = np.eye(len(w)) / C
    P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Zi @ w))
        g = Zi.T @ (p - y) + P @ w
        H = Zi.T @ (Zi * (p * (1 - p))[:, None]) + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def own_cv(X, y, reps=10, seed0=777):
    res = []
    for r in range(reps):
        rg = np.random.default_rng(seed0 + r)
        fold = np.zeros(len(y), int)
        for c in (0, 1):
            idx = np.where(y == c)[0]
            rg.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % 5
        oof = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            med = np.nanmedian(X[tr], 0)
            a, b = np.where(np.isnan(X[tr]), med, X[tr]), np.where(np.isnan(X[te]), med, X[te])
            mu, sd = a.mean(0), a.std(0)
            w = logit_fit((a - mu) / sd, y[tr])
            oof[te] = np.c_[np.ones(te.sum()), (b - mu) / sd] @ w
        res.append(own_auc(y, oof))
    return np.array(res)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    recs = {}
    for line in open(PRIV / "verbatim_20260929" / "verbatim_transcripts.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["status"] == "succeeded":
            recs[r["sample_id"]] = [(w["start"], w["end"], w["word"]) for w in r["words"]]
    rows = json.load(open(LP.OUT / "disfl_clip_rows.json", encoding="utf-8"))
    win = {r["clip_id"]: (float(r["start_s"]), float(r["end_s"]))
           for r in csv.DictReader(open(PRIV / "listening_20260926" / "listening_key_DO_NOT_OPEN.csv", encoding="utf-8-sig"))
           if r["repeat_of"] == ""}
    phon_whole = {r["sample_id"]: float(r["whole__phonation_s"]) / 60
                  for r in csv.DictReader(open(LP.RUN / "features_private.csv", encoding="utf-8-sig"))}
    gb = [r for r in rows if r["j"] in ("好", "不好")]
    y = np.array([1 if r["j"] == "好" else 0 for r in gb])
    bad = 1 - y

    # phonation minutes per clip: pause route; then (all-65 version) the filled-pause route n_fp / D1; then the median
    direct = [r["det"]["n_pause"] / r["p5"][1] if r["det"]["n_pause"] > 0 and r["p5"][1] > 0 else np.nan for r in gb]
    routed = [d if not np.isnan(d) else (r["det"]["n_fp"] / r["d"][0] if r["det"]["n_fp"] > 0 and r["d"][0] > 0 else np.nan)
              for r, d in zip(gb, direct)]
    med = float(np.nanmedian(routed))
    counts = [own_counts(recs[r["sid"]], win[r["clip"]]) for r in gb]

    def clip_cols(ph):
        return np.array([(f / p, rep / p) for (f, rep), p in zip(counts, ph)], float)

    clips = {"clip_all65": clip_cols([p if not np.isnan(p) else med for p in routed]), "clip_first_run63": clip_cols(direct)}
    whole = np.array([[c / phon_whole[r["sid"]] for c in own_counts(recs[r["sid"]])] for r in gb], float)
    out = {"n": len(gb), "n_bad": int(bad.sum())}
    for label, cols in (*clips.items(), ("whole_recording", whole)):
        for k, name in enumerate(("D4_filler_rate", "D5_repetition_rate")):
            v = cols[:, k]
            ok = ~np.isnan(v)
            u = mannwhitneyu(v[ok][bad[ok] == 1], v[ok][bad[ok] == 0], alternative="greater")
            auc_mw = u.statistic / (bad[ok].sum() * (1 - bad[ok]).sum())
            auc, p = own_perm_p(v, bad)
            out[f"{label}_{name}"] = {"auroc_mannwhitney": round(float(auc_mw), 3), "auroc_own": round(auc, 3),
                                      "p_one_sided_own_permutation": round(p, 4), "n": int(ok.sum())}
    A = np.load(LP.OUT / "disfl_main_arrays.npz")
    base = np.c_[A["X5"], A["XD"]]
    a_base = own_cv(base, y)
    out["own_cv_auroc_P5_D2_D3"] = round(float(a_base.mean()), 3)
    for label, cols in clips.items():
        a_full = own_cv(np.c_[base, cols], y)
        rng, n_perm = np.random.default_rng(20260928), 199
        obs = float((a_full - a_base).mean())
        null = [(own_cv(np.c_[base, cols[rng.permutation(len(y))]], y) - a_base).mean() for _ in range(n_perm)]
        out[f"{label}_own_cv"] = {"auroc_plus_D4_D5": round(float(a_full.mean()), 3), "delta": round(obs, 3),
                                  "perm_p_one_sided_199_perms": round(float((1 + np.sum(np.array(null) >= obs)) / (n_perm + 1)), 3)}
    out_dir = LP.PUBLIC / "verbatim"
    json.dump(out, open(out_dir / "verify.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
