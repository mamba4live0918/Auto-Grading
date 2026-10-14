"""Station 7 confirmatory analysis (plan: docs/research/2026-09-29-verbatim-plan.md, section 2).

Run only after verbatim_check.py has passed the uh pass line. Same conventions as station 6 (disfl_main.py):
L2 logistic regression C=1, 10 x stratified 5-fold CV (seed 20260928 + r), median imputation, 999 permutations.
Main test 1 uses seed 20260929 as the plan says.

The plan tests all 65 listened clips. The first run left out two clips whose phonation time could not be recovered from
the saved station 6 rows (no silent pause in 30 s); both are "not good" clips full of fillers, so leaving them out
biased the tests (plan deviation 3). Now one is recovered exactly through its filled-pause count and one gets the median
phonation time; the result does not depend on that guess. The first-run version is kept next to the main one.

Run: .venv311\\Scripts\\python.exe scripts\\acoustic_redo\\verbatim_main.py
Numbers without recording ids go to output/verbatim_20260929/main.json; per-clip rows stay in data/private/.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import verbatim as VB
import verbatim_check as C

import localpaths as LP
ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "data" / "private"
CLIP_ROWS = LP.OUT / "disfl_clip_rows.json"
MAIN_ARRAYS = LP.OUT / "disfl_main_arrays.npz"
LISTEN_KEY = PRIV / "listening_20260926" / "listening_key_DO_NOT_OPEN.csv"
FEATURES = LP.RUN / "features_private.csv"
OUT = LP.PUBLIC / "verbatim"
SEED_AUC, SEED_CV = 20260929, 20260928
N_PERM, REPS, FOLDS = 999, 10, 5
TIE_TOL = 1e-12  # permuted and observed AUROCs that tie mathematically can differ by one float step


def require_uh_pass(check):
    """The plan drops D4 when the uh validation fails; this script only runs after it passed."""
    if not check.get("uh_passed"):
        raise SystemExit("the uh pass line was not met: D4 does not enter the main tests (plan section 1)")


def _need_both_classes(labels):
    if len(np.unique(labels)) < 2:
        raise ValueError("AUROC is undefined: the labels do not contain both classes")


def holm(pvals):
    """Holm step-down adjusted p-values for {name: p}."""
    order = sorted(pvals, key=pvals.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[name]))
        adjusted[name] = running
    return adjusted


def auroc_bad(values, bad):
    """AUROC for picking out the 'bad' clips when a higher value means worse; NaN values are left out."""
    v, b = np.asarray(values, float), np.asarray(bad)
    ok = ~np.isnan(v)
    _need_both_classes(b[ok])
    return float(roc_auc_score(b[ok], v[ok]))


def perm_p_auc(values, bad, seed=SEED_AUC, n_perm=N_PERM):
    """One-sided permutation p for auroc_bad."""
    v, b = np.asarray(values, float), np.asarray(bad)
    ok = ~np.isnan(v)
    v, b = v[ok], b[ok]
    _need_both_classes(b)
    observed = roc_auc_score(b, v)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(b), v) for _ in range(n_perm)])
    return float((1 + np.sum(null >= observed - TIE_TOL)) / (n_perm + 1))


def phonation_minutes(det, p5):
    """Phonation minutes of a clip, recovered from its pause count and pause rate (P5 order: rate is item 1)."""
    rate = p5[1]
    if det["n_pause"] > 0 and rate and rate > 0:
        return det["n_pause"] / rate
    return None


def phonation_minutes_fallback(det, p5, d):
    """Like phonation_minutes, but a clip without a silent pause can also be recovered from its filled-pause count (n_fp / D1)."""
    ph = phonation_minutes(det, p5)
    if ph is None and det["n_fp"] > 0 and d[0] and d[0] > 0:
        return det["n_fp"] / d[0]
    return ph


def clip_rates(words, window, ph_min):
    """(D4 filler rate, D5 repetition rate) per phonation minute inside `window`; NaN without a usable time."""
    if not ph_min or ph_min <= 0:
        return float("nan"), float("nan")
    return len(VB.filler_spans(words, window)) / ph_min, len(VB.repetition_spans(words, window)) / ph_min


def _impute(train, test):
    med = np.nanmedian(train, 0)
    return np.where(np.isnan(train), med, train), np.where(np.isnan(test), med, test)


def cv_auc(X, y, reps=REPS):
    """Cross-validated AUROC of an L2 logistic regression, one value per repeat."""
    aucs = []
    for r in range(reps):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED_CV + r).split(X, y):
            a, b = _impute(X[tr], X[te])
            scaler = StandardScaler().fit(a)
            model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(a), y[tr])
            oof[te] = model.predict_proba(scaler.transform(b))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return np.array(aucs)


def perm_p_delta(base, extra, y, observed_delta, n_perm=N_PERM, reps=REPS):
    """One-sided p for the mean CV-AUROC gain from adding `extra`: the added columns are shuffled together across recordings."""
    rng = np.random.default_rng(SEED_CV)
    a_base = cv_auc(base, y, reps)
    null = np.array([(cv_auc(np.c_[base, extra[rng.permutation(len(y))]], y, reps) - a_base).mean() for _ in range(n_perm)])
    return float((1 + np.sum(null >= observed_delta - TIE_TOL)) / (n_perm + 1))


def _test1(columns, bad):
    """Main test 1 for a {name: values} dict: AUROC, permutation p, and Holm across the names."""
    res = {name: {"auroc_predicting_bad": round(auroc_bad(v, bad), 3), "p_one_sided": round(perm_p_auc(v, bad), 4),
                  "n": int((~np.isnan(np.asarray(v, float))).sum())} for name, v in columns.items()}
    adjusted = holm({k: r["p_one_sided"] for k, r in res.items()})
    for k in res:
        res[k]["p_holm"] = round(adjusted[k], 4)
    return res


def _columns(matrix):
    return {"D4_filler_rate": matrix[:, 0], "D5_repetition_rate": matrix[:, 1]}


def _model_test(base, extra, y, a_base):
    """Main test 2: CV-AUROC of base + extra against base alone, with the permutation p of the mean gain."""
    gain = cv_auc(np.c_[base, extra], y) - a_base
    return {"auroc_P5_D2_D3": round(float(a_base.mean()), 3), "auroc_plus_D4_D5": round(float((a_base + gain).mean()), 3),
            "delta_mean": round(float(gain.mean()), 3), "delta_range": [round(float(gain.min()), 3), round(float(gain.max()), 3)],
            "perm_p_one_sided": round(perm_p_delta(base, extra, y, float(gain.mean())), 4)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require_uh_pass(json.load(open(C.OUT / "check1.json", encoding="utf-8")))
    records = C.load_new_records(C.NEW)
    C.check_provenance(list(records.values()))
    new = {sid: C.words_of_record(r) for sid, r in records.items()}
    rows = json.load(open(CLIP_ROWS, encoding="utf-8"))
    windows = {r["clip_id"]: (float(r["start_s"]), float(r["end_s"]))
               for r in csv.DictReader(open(LISTEN_KEY, encoding="utf-8-sig")) if r["repeat_of"] == ""}

    gb = [x for x in rows if x["j"] in ("好", "不好")]
    y = np.array([1 if x["j"] == "好" else 0 for x in gb])
    bad = 1 - y
    X5 = np.array([x["p5"] for x in gb], float)
    XD = np.array([[x["d"][1], x["d"][2]] for x in gb], float)

    # the data assembly must reproduce station 6 exactly before anything new is computed
    ref = np.load(MAIN_ARRAYS)
    if not (np.array_equal(ref["y"], y) and np.allclose(ref["X5"], X5, equal_nan=True) and np.allclose(ref["XD"], XD, equal_nan=True)):
        raise SystemExit("station 6 inputs are not reproduced by this assembly")
    a5, a5d = cv_auc(X5, y), cv_auc(np.c_[X5, XD], y)
    if not (np.allclose(a5, ref["a5"]) and np.allclose(a5d, ref["a5d"])):
        raise SystemExit("station 6 cross-validated AUROCs are not reproduced by cv_auc")

    # phonation time per clip: the pause route (first run), then the filled-pause route, then the median for what is left
    for x in rows:
        x["ph_direct"] = phonation_minutes(x["det"], x["p5"])
        x["ph_routed"] = phonation_minutes_fallback(x["det"], x["p5"], x["d"])
    known = np.array([x["ph_routed"] for x in gb if x["ph_routed"]])
    median_ph = float(np.median(known))
    for x in rows:
        x["ph_min"] = x["ph_routed"] or median_ph
        x["D4"], x["D5"] = clip_rates(new[x["sid"]], windows[x["clip"]], x["ph_min"])
    XN = np.array([[x["D4"], x["D5"]] for x in gb], float)
    first_run = np.array([clip_rates(new[x["sid"]], windows[x["clip"]], x["ph_direct"]) for x in gb], float)

    out = {"n_clips": len(gb), "n_good": int(y.sum()),
           "station6_reproduced": {"auroc_P5": round(float(a5.mean()), 3), "auroc_P5_D2_D3": round(float(a5d.mean()), 3)},
           "clips_without_a_silent_pause": {
               "total": sum(1 for x in gb if x["ph_direct"] is None),
               "recovered_by_filled_pause_route": sum(1 for x in gb if x["ph_direct"] is None and x["ph_routed"]),
               "given_the_median_phonation_time": sum(1 for x in gb if not x["ph_routed"])}}
    out["main1_clip"] = _test1(_columns(XN), bad)
    out["main1_clip_first_run_63_clips"] = _test1(_columns(first_run), bad)
    base = np.c_[X5, XD]
    out["main2_clip"] = _model_test(base, XN, y, a5d)
    out["main2_clip_first_run"] = _model_test(base, first_run, y, a5d)
    swept = {}
    for guess in (known.min(), median_ph, known.max()):
        cols = np.array([clip_rates(new[x["sid"]], windows[x["clip"]], x["ph_routed"] or guess) for x in gb], float)
        swept[f"{guess:.2f}_min"] = {"D4_auroc": round(auroc_bad(cols[:, 0], bad), 3), "D5_auroc": round(auroc_bad(cols[:, 1], bad), 3),
                                     "delta_mean": round(float((cv_auc(np.c_[base, cols], y) - a5d).mean()), 3)}
    out["if_the_one_guessed_phonation_time_were"] = swept

    # whole recordings (both windows are reported): D4 / D5 over the full transcript, Praat phonation time from the features file
    phon = {r["sample_id"]: float(r["whole__phonation_s"]) / 60 for r in csv.DictReader(open(FEATURES, encoding="utf-8-sig"))}
    whole = np.array([[len(VB.filler_spans(new[x["sid"]])) / phon[x["sid"]], len(VB.repetition_spans(new[x["sid"]])) / phon[x["sid"]]]
                      for x in gb], float)
    out["main1_whole_recording"] = {
        "note": "added after the frozen plan (report both windows); Holm over its own two tests only",
        **_test1(_columns(whole), bad)}

    note = np.array([x["disfl_note"] for x in rows])
    out["descriptive_vs_listener_note"] = {name: round(float(roc_auc_score(note, [x[name] for x in rows])), 3) for name in ("D4", "D5")}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "main.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump([{k: x[k] for k in ("clip", "j", "ph_min", "D4", "D5")} for x in rows],
              open(LP.RUN / "verbatim_main_rows.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
