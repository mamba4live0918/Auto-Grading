"""E4 descriptives (9/28 output/listening_20260928/descriptive.json; the script that wrote it was not kept), recomputed
locally: teacher total score by listener label, single-indicator AUROCs (30-s clips and whole recordings, raw
direction: higher value = "good"), and the keyword tally of the listener's notes. The keyword lists were rebuilt so
that they reproduce the 9/28 tally; the comparison is printed and the run stops if anything differs.
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/listening_descriptive_local.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

import localpaths as LP

P5 = ["articulation_rate", "pause_rate_per_min", "mean_pause_s", "f0_sd_st", "intensity_sd_db"]
THEMES = {  # theme -> keywords (any hit counts the note once)
    "卡顿/uh/重复": ["卡顿", "uh", "重复"],
    "太平/没语调": ["太平", "有点平"],
    "像读稿": ["读稿", "在读", "念"],
    "杂音/人声不清": ["杂音", "人声", "背景音"],
    "不自信": ["不自信"],
    "准备不足": ["没准备", "没有准备", "没提前准备"],
}


def main() -> None:
    key = [r for r in csv.DictReader(open(LP.LISTEN_KEY, encoding="utf-8")) if r["repeat_of"] == ""]
    lab = {r["clip_id"]: r for r in csv.DictReader(open(LP.LISTEN_LABELS, encoding="utf-8-sig"))}
    with open(LP.MAN, encoding="utf-8") as f:
        score = {r["sample_id"]: float(r["total_score"]) for r in csv.DictReader(f) if r["score_status"] == "observed"}
    feats = {r["sample_id"]: r for r in csv.DictReader(open(LP.RUN / "features_private.csv", encoding="utf-8-sig"))}
    clip_p5 = {r["clip"]: r["p5"] for r in json.load(open(LP.OUT / "disfl_clip_rows.json", encoding="utf-8"))}

    rows = [{"clip": r["clip_id"], "sid": r["sample_id"], "j": lab[r["clip_id"]]["judgement"],
             "note": lab[r["clip_id"]]["note"]} for r in key]
    out = {}
    for j in ("好", "说不准", "不好"):
        s = [score[r["sid"]] for r in rows if r["j"] == j]
        out[f"score_{j}"] = {"n": len(s), "mean": round(float(np.mean(s)), 2), "median": round(float(np.median(s)), 2)}
    gb = [r for r in rows if r["j"] in ("好", "不好")]
    y = np.array([r["j"] == "好" for r in gb], int)
    sc = np.array([score[r["sid"]] for r in gb])
    out["teacher_score_auroc_good_vs_bad"] = round(float(roc_auc_score(y, sc)), 3)
    out["mw_p"] = float(mannwhitneyu(sc[y == 1], sc[y == 0]).pvalue)
    level = {"好": 2, "说不准": 1, "不好": 0}
    r3, p3 = spearmanr([level[r["j"]] for r in rows], [score[r["sid"]] for r in rows])
    out["spearman_label3_vs_score"] = [round(float(r3), 3), round(float(p3), 3)]

    # missing values (no pause at all -> no mean pause duration) take the median of the array they come from, as on
    # 9/28: the 82 recordings for the whole-recording version, the 65 judged clips for the 30-s version
    def aurocs(values: np.ndarray, fill_from: np.ndarray) -> dict:
        res = {}
        for i, k in enumerate(P5):
            v = np.where(np.isnan(values[:, i]), np.nanmedian(fill_from[:, i]), values[:, i])
            res[k] = round(float(roc_auc_score(y, v)), 3)
        return res

    def whole(sid: str) -> list[float]:
        return [float(feats[sid][f"whole__{k}"]) if feats[sid][f"whole__{k}"] else np.nan for k in P5]

    W = np.array([whole(r["sid"]) for r in gb], float)
    C = np.array([[np.nan if v is None else v for v in clip_p5[r["clip"]]] for r in gb], float)
    out["whole_feature_auroc"] = aurocs(W, np.array([whole(s) for s in feats], float))
    out["posthoc_clip_feature_auroc"] = aurocs(C, C)
    out["note_themes"] = {t: {j: sum(1 for r in rows if r["j"] == j and any(w in r["note"].lower() for w in kws))
                              for j in ("好", "说不准", "不好")} for t, kws in THEMES.items()}
    out["note_theme_keywords"] = THEMES

    old = json.load(open(LP.ROOT / "output/listening_20260928/descriptive.json", encoding="utf-8"))
    diffs = []
    for k, v in old.items():
        new = out.get(k)
        if isinstance(v, float):
            if abs(v - new) > 1e-9:
                diffs.append((k, v, new))
        elif v != new:
            diffs.append((k, v, new))
    (LP.PUBLIC / "listening_descriptive.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("differences from 9/28:", diffs if diffs else "none")
    if diffs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
