"""E1 post hoc (deviation 2 of the 9/26 plan): Ridge on the five primary indicators measured on the first ~48 s of
phonation, same folds and nested ridge as the main analysis. The 9/26 one-off script was not kept; this is the same
calculation (model.run with the first48 features) and reproduces the saved MAE.
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/posthoc_first48_local.py
"""
from __future__ import annotations

import json

import numpy as np

import localpaths as LP
import model as M
import analysis_inputs as AI

P5 = ["articulation_rate", "pause_rate_per_min", "mean_pause_s", "f0_sd_st", "intensity_sd_db"]


def main() -> None:
    rec = {i: json.load(open(LP.OUT / f"{i}.json")) for i in AI.ids}
    X48 = np.array([[np.nan if rec[i]["first48"][k] is None else rec[i]["first48"][k] for k in P5] for i in AI.ids], float)
    y = np.load(LP.OUT / "preds.npz")["y"]
    preds, _ = M.run({"B0": ("median", None), "R48": ("ridge", X48)}, y, M.outer_splits(len(y)))
    res = {"posthoc_Ridge_P5_first48_mae": float(M.mae(preds["R48"], y).mean()), "B0": float(M.mae(preds["B0"], y).mean())}
    (LP.OUT / "posthoc_first48.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)


if __name__ == "__main__":
    main()
