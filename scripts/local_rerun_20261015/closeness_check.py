"""Closeness-to-the-professional-profile check (plan: docs/research/2026-09-29-closeness-plan.md).

Intent (user, 2026-09-29): the professional speakers define a standard of "good"; a student who is
closer to it is better. The 9/26 analysis instead trained a classifier (professional vs student) and used its
probability as "professional-likeness", which answers "which side does this look like", not "how far from
the standard". Here the standard is the professionals' mean and spread of the 5 main indicators, and the
distance is the root of the summed squared z-scores. Runs locally on data already in data/private.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import localpaths as LP
ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "data" / "private"
PROFESSIONALS = LP.RUN / "professional_features.csv"
STUDENT_FEATURES = LP.RUN / "features_private.csv"
GROUPS = LP.RUN / "students_ref.npz"
LISTEN_LABELS = PRIV / "listening_20260926" / "听辨结果_2026-09-28.csv"
LISTEN_KEY = PRIV / "listening_20260926" / "listening_key_DO_NOT_OPEN.csv"
MANIFEST = (ROOT / ".worktrees" / "europarl-500" / "artifacts" / "research" /
            "444eb952b40469ad9e3dfe8983ea7461335f89f23ca0ed99de465d20d55c10ca" / "manifest.csv")
OUT = LP.PUBLIC / "closeness_check"
P5 = ["articulation_rate", "pause_rate_per_min", "mean_pause_s", "f0_sd_st", "intensity_sd_db"]
SEED, N_PERM = 20260929, 999


def to_float(text: str) -> float:
    return float(text) if text.strip() else float("nan")


def load() -> dict:
    with open(PROFESSIONALS, encoding="utf-8") as f:
        pros = np.array([[to_float(r[k]) for k in P5] for r in csv.DictReader(f)])
    ids = [str(i) for i in np.load(GROUPS)["ids"]]
    with open(STUDENT_FEATURES, encoding="utf-8") as f:
        feats = {r["sample_id"]: r for r in csv.DictReader(f)}
    stu = np.array([[to_float(feats[i][f"whole__{k}"]) for k in P5] for i in ids])
    stu = np.where(np.isnan(stu), np.nanmedian(stu, axis=0), stu)
    with open(LISTEN_LABELS, encoding="utf-8-sig") as f:
        judged = {r["clip_id"]: r["judgement"] for r in csv.DictReader(f)}
    with open(LISTEN_KEY, encoding="utf-8") as f:
        first = {r["sample_id"]: judged[r["clip_id"]] for r in csv.DictReader(f) if r["repeat_of"] == ""}
    with open(MANIFEST, encoding="utf-8") as f:
        score = {r["sample_id"]: float(r["total_score"]) for r in csv.DictReader(f)
                 if r["score_status"] == "observed" and r["supervised_eligible"] == "True"}
    grp = np.load(GROUPS)
    return {"ids": ids, "pros": pros, "stu": stu, "judged": [first[i] for i in ids],
            "score": np.array([score[i] for i in ids]), "group": grp["group"], "plike": grp["plike"]}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    ranks = rankdata(s)
    n1 = int(y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def perm_p(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> float:
    observed = auroc(y, s)
    hits = sum(auroc(rng.permutation(y), s) >= observed for _ in range(N_PERM))
    return (1 + hits) / (1 + N_PERM)


def euclid_distance(stu: np.ndarray, pros: np.ndarray) -> np.ndarray:
    z = (stu - pros.mean(0)) / pros.std(0, ddof=1)
    return np.sqrt((z ** 2).sum(1))


def mahal_distance(stu: np.ndarray, pros: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.cov(pros, rowvar=False))
    diff = stu - pros.mean(0)
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def main() -> None:
    d = load()
    rng = np.random.default_rng(SEED)
    stu, pros, score = d["stu"], d["pros"], d["score"]
    dist_e, dist_m = euclid_distance(stu, pros), mahal_distance(stu, pros)

    inv = np.linalg.inv(np.cov(pros, rowvar=False))
    recomputed_e = np.array([euclidean((s - pros.mean(0)) / pros.std(0, ddof=1), np.zeros(5)) for s in stu])
    recomputed_m = np.array([mahalanobis(s, pros.mean(0), inv) for s in stu])
    assert np.allclose(dist_e, recomputed_e) and np.allclose(dist_m, recomputed_m), "distance recompute disagrees"

    judged = np.array(d["judged"])
    keep = np.isin(judged, ["好", "不好"])
    y = (judged[keep] == "好").astype(int)
    out = {"n_students": len(stu), "n_good_bad": int(keep.sum()), "n_good": int(y.sum()),
           "professional_mean": np.round(pros.mean(0), 3).tolist(), "professional_sd": np.round(pros.std(0, ddof=1), 3).tolist()}

    primary = auroc(y, -dist_e[keep])
    assert abs(primary - roc_auc_score(y, -dist_e[keep])) < 1e-12
    out["primary_euclidean"] = {"auroc_closer_means_good": round(primary, 3),
                                "perm_p_one_sided": round(perm_p(y, -dist_e[keep], rng), 3)}
    out["descriptive"] = {
        "mahalanobis_auroc": round(auroc(y, -dist_m[keep]), 3),
        "previous_classifier_probability_auroc": round(auroc(y, d["plike"][keep]), 3),
        "spearman_closeness_vs_total_score": [round(float(spearmanr(-dist_e, score)[0]), 3), round(float(spearmanr(-dist_e, score)[1]), 3)],
        "auroc_above_mean_score": round(auroc((score > score.mean()).astype(int), -dist_e), 3),
        "mean_distance_by_group": {str(g): round(float(dist_e[d["group"] == g].mean()), 2) for g in sorted(set(d["group"].tolist()))},
        "per_indicator_auroc_closer_means_good": {
            k: round(auroc(y, -np.abs((stu[keep, j] - pros[:, j].mean()) / pros[:, j].std(ddof=1))), 3) for j, k in enumerate(P5)},
        "distance_median_and_range": [round(float(np.median(dist_e)), 2), round(float(dist_e.min()), 2), round(float(dist_e.max()), 2)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
