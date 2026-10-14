"""事后、描述性的“能排除多大的效应”计算（2026-10-12，为论文第 4 章的“主要发现”）。

只读已保存的汇总结果，不读学生录音，不训练模型：
1. 两组“好”的比例差的 95% 区间（Newcombe 混合 Wilson 区间），输入是聚类分组 × 听者判断的 2×2 表；
2. 离专业标准的距离 AUROC = 0.56（40 好 / 25 不好）的 Hanley–McNeil 近似 95% 区间；
3. 正态近似下，预测与总分的相关 r 能让平均误差减少多少：MAE 比例 = sqrt(1 - r^2)；
4. 预测总分的 MAE 差值（候选模型 − B0）按 Nadeau–Bengio 校正方差得到的 95% 区间。
   用已保存的留出预测（data/private/acoustic_redo_20260926/preds.npz，只在本机读取）和与原分析相同的
   10 × 5 划分（scripts/acoustic_redo/model.py 的 outer_splits）重算每折的差值；先核对重算的 t 值与
   analysis.json 里保存的 nb_t 一致，再给区间。只输出汇总数字，不输出逐份录音的数据。
结果写到 output/findings_bounds_20261012/results.json。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

import localpaths as LP
ROOT = Path(__file__).resolve().parents[2]
OUT = LP.PUBLIC / "findings_bounds"
PREDS = LP.OUT / "preds.npz"
Z = 1.959963984540054


def nadeau_bengio_intervals(saved_contrasts: list[dict]) -> dict:
    """每个候选模型对 B0 的 MAE 差值：重算 Nadeau–Bengio t，核对后给 95% 区间。"""
    sys.path.insert(0, str(ROOT / "scripts" / "acoustic_redo"))
    import model as M  # noqa: E402  与原分析同一份划分代码

    data = np.load(PREDS)
    y = data["y"]
    n = len(y)
    splits = M.outer_splits(n)
    ratio = (n / M.K) / (n - n / M.K)
    saved = {c["candidate"]: c for c in saved_contrasts if c["baseline"] == "B0_median"}
    out = {}
    for name in ["Ridge_P5", "Ridge_E", "Ridge_G"]:
        d = np.abs(data[name] - y) - np.abs(data["B0_median"] - y)  # (R, n)
        dj = np.array([d[r, te].mean() for r, folds in enumerate(splits) for _tr, te in folds])
        J = len(dj)
        se = math.sqrt((1 / J + ratio) * dj.var(ddof=1))
        t_obs = dj.mean() / se
        if not math.isclose(t_obs, saved[name]["nb_t"], rel_tol=1e-9, abs_tol=1e-12):
            raise SystemExit(f"{name}: recomputed nb_t {t_obs} != saved {saved[name]['nb_t']}")
        q = float(tdist.ppf(0.975, J - 1))
        out[name] = {"mean_fold_diff": float(dj.mean()), "se": se, "t": t_obs, "df": J - 1,
                     "ci95": [float(dj.mean() - q * se), float(dj.mean() + q * se)],
                     "saved_mae_diff": saved[name]["mae_diff"], "saved_boot95": saved[name]["boot95"]}
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> dict:
    """p1 - p2 的 95% 区间（Newcombe 1998 的方法 10）。"""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {"p1": p1, "p2": p2, "diff": d, "ci95": [lower, upper]}


def hanley_mcneil(auc: float, n_pos: int, n_neg: int) -> dict:
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    se = math.sqrt(var)
    return {"auc": auc, "se": se, "ci95": [auc - Z * se, auc + Z * se]}


def mae_reduction(r: float, baseline_mae: float) -> dict:
    ratio = math.sqrt(1 - r * r)
    return {"r": r, "relative_reduction": 1 - ratio, "points_at_baseline": baseline_mae * (1 - ratio)}


def main() -> None:
    primary = json.loads((LP.OUT / "label_analysis.json").read_text(encoding="utf-8"))
    (g1_good, g1_bad), (g2_good, g2_bad) = primary["group_table_good_bad"]
    redo = json.loads((LP.OUT / "analysis.json").read_text(encoding="utf-8"))
    b0 = next(m["mae"] for m in redo["models"] if m["model"] == "B0_median")
    corr = json.loads((LP.PUBLIC / "report_correlations" / "results.json").read_text(encoding="utf-8"))
    r_index = corr["new_recording_index_vs_teacher"]["pearson"]["coefficient"]
    r_ridge = corr["ridge_p5_saved_oof_vs_teacher"]["mean_oof_prediction_correlations"]["pearson"]["coefficient"]

    result = {
        "status": "post hoc, descriptive; summary numbers only; no student audio, no model fitting",
        "group_good_proportion_difference": {
            "table_good_bad": [[g1_good, g1_bad], [g2_good, g2_bad]],
            "method": "Newcombe hybrid Wilson score interval for p1 - p2",
            **newcombe_diff(g1_good, g1_good + g1_bad, g2_good, g2_good + g2_bad),
        },
        "distance_auroc_interval": {
            "method": "Hanley-McNeil normal approximation; 40 good / 25 bad",
            **hanley_mcneil(round(json.loads((LP.PUBLIC / "closeness_check" / "results.json").read_text(encoding="utf-8"))["primary_euclidean"]["auroc_closer_means_good"], 2), 40, 25),
        },
        "nadeau_bengio_mae_diff_intervals": {
            "method": "fold-level MAE differences vs B0, corrected variance (1/J + n_test/n_train), t with J-1 df; "
                      "recomputed t checked against analysis.json nb_t",
            **nadeau_bengio_intervals(redo["contrasts"]),
        },
        "mae_reduction_normal_approximation": {
            "formula": "MAE ratio = sqrt(1 - r^2) for bivariate normal data and a calibrated linear prediction",
            "baseline_mae_B0": b0,
            "index_pearson_r": mae_reduction(r_index, b0),
            "ridge_p5_mean_oof_pearson_r": mae_reduction(r_ridge, b0),
        },
        "inputs": {
            "listening_primary": "data/private/local_rerun_20261015/out/label_analysis.json",
            "acoustic_redo": "data/private/local_rerun_20261015/out/analysis.json",
            "report_correlations": "output/local_rerun_20261015/report_correlations/results.json",
            "distance_auroc": "output/local_rerun_20261015/closeness_check/results.json",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
