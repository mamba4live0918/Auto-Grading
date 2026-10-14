"""The 9/30 descriptive diagnostic "change only the transcript that decides pause positions" (results chapter,
model paragraph), recomputed: same 65 clips, same folds (pause_remeasure cv_auc), P5 + long and mid-sentence pause
rates [+ filler rate] [+ repetition rate], once per transcript at the 250-ms threshold.

First recomputes it from the 9/30 inputs and checks that the saved 9/30 numbers come out (so this is the same
calculation), then runs it on the local re-run inputs.
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/transcript_model_diagnostic.py
"""
from __future__ import annotations

import json

import numpy as np

import localpaths as LP
import pause_remeasure_20260930 as PR

CONDITIONS = {"old_whisper": "250_old_whisper", "original": "250_original", "denoised": "250_denoised", "crisper": "250_crisper"}


def diagnostic(rows_path, verbatim_rows_path) -> dict:
    rows = json.load(open(rows_path, encoding="utf-8"))
    # repetition count per clip (E12) divided by the phonation time measured with the pauses (as on 9/30)
    rep = {r["clip"]: round(r["D5"] * r["ph_min"]) for r in json.load(open(verbatim_rows_path, encoding="utf-8"))}
    gb = [r for r in rows if r["j"] in ("好", "不好")]
    assert len(gb) == 65
    y = np.array([r["j"] == "好" for r in gb], int)
    X5 = np.array([r["p5"] for r in gb], float)
    out = {}
    for name, key in CONDITIONS.items():
        m = PR.condition_matrix(gb, key)
        base = np.c_[X5, m["long_rate"], m["mid_rate"]]
        fill = np.asarray(m["filler_rate"], float)[:, None]
        D5 = np.array([rep[r["clip"]] / (r["conditions"][key]["phonation_s"] / 60) for r in gb], float)[:, None]
        out[name] = {"P5_plus_long_mid": round(float(PR.cv_auc(base, y).mean()), 4),
                     "P5_plus_long_mid_filler": round(float(PR.cv_auc(np.c_[base, fill], y).mean()), 4),
                     "P5_plus_long_mid_filler_repeat": round(float(PR.cv_auc(np.c_[base, fill, D5], y).mean()), 4)}
    return out


def main() -> None:
    saved = json.load(open(LP.ROOT / "output/pause_remeasure_20260930/model_input_diagnostic.json", encoding="utf-8"))
    old = diagnostic(LP.PRIV / "pause_remeasure_20260930/assembled_rows.json", LP.PRIV / "verbatim_20260929/main_rows.json")
    target = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in saved["change_only_pause_label_transcript"].items()}
    print("9/30 calculation reproduced:", old == target)
    if old != target:
        print(json.dumps({"saved": target, "recomputed": old}, indent=1))
        raise SystemExit("the diagnostic is not the same calculation; stop")
    new = diagnostic(LP.RUN / "pause_remeasure/assembled_rows.json", LP.RUN / "verbatim_main_rows.json")
    (LP.PUBLIC / "transcript_model_diagnostic.json").write_text(json.dumps(
        {"n": 65, "change_only_pause_label_transcript": new, "same_as_2026_09_30": new == old}, indent=1), encoding="utf-8")
    print(json.dumps(new, indent=1), "\nsame as 9/30:", new == old)


if __name__ == "__main__":
    main()
