"""Post hoc check (2026-09-29, added after seeing the listening results).

What if "good" means "teacher total score above the mean" instead of the listener's judgement?
Repeats the three checks that used the listener's labels, on all 82 recordings:
  A. Do the 9/26 machine groups (Ward on the 5 main indicators, whole recording) differ in the
     share of above-mean recordings?
  B. Does "similar to professional speakers" tell above-mean from at-or-below-mean?
  C. Do the 5 main indicators tell them apart (L2 logistic regression, C = 1,
     10 x 5-fold stratified CV, imputation and scaling inside each training fold)?

Not part of any frozen plan: exploratory. Many analyses have already been run on the same 82
recordings, and dichotomising a score at its mean is an arbitrary cut. Runs locally; it only reads
data/private and the frozen manifest.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisher_exact, mannwhitneyu, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import localpaths as LP
ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "data" / "private"
MANIFEST = (ROOT / ".worktrees" / "europarl-500" / "artifacts" / "research" /
            "444eb952b40469ad9e3dfe8983ea7461335f89f23ca0ed99de465d20d55c10ca" / "manifest.csv")
FEATURES = LP.RUN / "features_private.csv"
GROUPS = LP.RUN / "students_ref.npz"
GROUPS_FULL = LP.RUN / "out_students_ref_full.npz"
OUT = LP.PUBLIC / "teacher_score_check"
P5 = ["articulation_rate", "pause_rate_per_min", "mean_pause_s", "f0_sd_st", "intensity_sd_db"]
SEED, N_PERM, REPEATS, FOLDS = 20260929, 999, 10, 5


def to_float(text: str) -> float:
    return float(text) if text.strip() else float("nan")


def load_inputs() -> dict:
    grp, full = np.load(GROUPS), np.load(GROUPS_FULL)
    ids = [str(i) for i in grp["ids"]]
    assert ids == [str(i) for i in full["ids"]], "student order differs between the two group files"
    with open(MANIFEST, encoding="utf-8") as f:
        score = {r["sample_id"]: float(r["total_score"]) for r in csv.DictReader(f)
                 if r["score_status"] == "observed" and r["supervised_eligible"] == "True"}
    with open(FEATURES, encoding="utf-8") as f:
        feats = {r["sample_id"]: r for r in csv.DictReader(f)}
    x = np.array([[to_float(feats[i][f"whole__{k}"]) for k in P5] for i in ids])
    return {"ids": ids, "x": x, "score": np.array([score[i] for i in ids]),
            "group": grp["group"], "group_full": full["group"],
            "plike": grp["plike"], "plike_full": full["plike"]}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (average ranks for ties); equals sklearn's roc_auc_score."""
    ranks = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def perm_p_score(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    observed = auroc(y, s)
    hits = sum(auroc(rng.permutation(y), s) >= observed for _ in range(N_PERM))
    return observed, (1 + hits) / (1 + N_PERM)


def cv_auroc(x: np.ndarray, y: np.ndarray) -> float:
    """Mean over REPEATS of the AUROC of out-of-fold predictions (scikit-learn pipeline)."""
    aucs = []
    for r in range(REPEATS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r).split(x, y):
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=5000))
            oof[te] = pipe.fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
        aucs.append(auroc(y, oof))
    return float(np.mean(aucs))


def fit_logreg_scipy(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> tuple[np.ndarray, float]:
    """Independent L2 logistic regression: min 0.5*|w|^2 + C*sum(log(1+exp(-s*(xw+b))))."""
    sign, n_feat = 2.0 * y - 1.0, x.shape[1]

    def objective(p: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = p[:n_feat], p[n_feat]
        z = sign * (x @ w + b)
        weight = c * sign * expit(-z)
        loss = 0.5 * w @ w + c * np.sum(np.logaddexp(0.0, -z))
        return loss, np.concatenate([w - x.T @ weight, [-weight.sum()]])

    res = minimize(objective, np.zeros(n_feat + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-10})
    return res.x[:n_feat], float(res.x[n_feat])


def cv_auroc_scratch(x: np.ndarray, y: np.ndarray) -> float:
    """Same folds and preprocessing as cv_auroc, but with the scipy-fitted model."""
    aucs = []
    for r in range(REPEATS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r).split(x, y):
            med = np.nanmedian(x[tr], axis=0)
            xtr, xte = np.where(np.isnan(x[tr]), med, x[tr]), np.where(np.isnan(x[te]), med, x[te])
            mu, sd = xtr.mean(0), xtr.std(0)
            sd = np.where(sd == 0, 1.0, sd)
            w, b = fit_logreg_scipy((xtr - mu) / sd, y[tr])
            oof[te] = ((xte - mu) / sd) @ w + b
        aucs.append(auroc(y, oof))
    return float(np.mean(aucs))


def perm_p_cv(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    hits = sum(cv_auroc(x, rng.permutation(y)) >= observed for _ in range(N_PERM))
    return (1 + hits) / (1 + N_PERM)


def group_check(y: np.ndarray, score: np.ndarray, group: np.ndarray) -> dict:
    labels = sorted(set(group.tolist()))
    table = [[int(((group == g) & (y == 1)).sum()), int(((group == g) & (y == 0)).sum())] for g in labels]
    a, b = score[group == labels[0]], score[group == labels[1]]
    return {"table_above_below": table, "n": [int((group == g).sum()) for g in labels],
            "share_above": [round(t[0] / sum(t), 3) for t in table],
            "fisher_p": round(float(fisher_exact(table)[1]), 3),
            "mean_score": [round(float(a.mean()), 1), round(float(b.mean()), 1)],
            "mannwhitney_p_raw_scores_descriptive": round(float(mannwhitneyu(a, b).pvalue), 3)}


def main() -> None:
    d = load_inputs()
    score, x, rng = d["score"], d["x"], np.random.default_rng(SEED)
    y = (score > score.mean()).astype(int)
    out = {"n": len(y), "mean_score": round(float(score.mean()), 2),
           "n_above_mean": int(y.sum()), "n_at_or_below_mean": int((1 - y).sum()),
           "groups_identical_in_both_files": bool(np.array_equal(d["group"], d["group_full"]))}
    out["A_groups"] = group_check(y, score, d["group"])
    for key, name in (("plike", "B_professional_similarity"), ("plike_full", "B_professional_similarity_full_length")):
        obs, p = perm_p_score(y, d[key], rng)
        assert abs(obs - roc_auc_score(y, d[key])) < 1e-12
        out[name] = {"auroc": round(obs, 3), "perm_p_one_sided": round(p, 3)}
    observed = cv_auroc(x, y)
    scratch = cv_auroc_scratch(x, y)
    out["C_five_indicators"] = {"cv_auroc": round(observed, 3), "perm_p_one_sided": round(perm_p_cv(x, y, observed, rng), 3),
                                "independent_recompute_auroc": round(scratch, 3),
                                "difference_vs_sklearn": round(abs(scratch - observed), 4)}
    assert abs(scratch - observed) < 0.005, "independent recompute disagrees with scikit-learn"
    out["single_indicator_auroc_higher_value_means_above"] = {
        k: round(auroc(y[~np.isnan(x[:, j])], x[~np.isnan(x[:, j]), j]), 3) for j, k in enumerate(P5)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
