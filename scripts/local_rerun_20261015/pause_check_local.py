"""E5 (pause check against the 24 annotated clips), locally.

The 9/28 key held, for each annotated 30-s clip, the program's pauses (Praat and WebRTC) measured in the cloud.
Here the same clip windows are kept and the program pauses are measured again from the local Praat/WebRTC
output, with the same rule (silent gaps of at least 0.25 s between sounding stretches inside the window, times
relative to the clip start); the 9/28 analysis script is then run unchanged on the new key. Also computes the
share of program pauses of at least 1 s among flagged and other pauses (reported in the 9/28 disfluency summary).

The key stores each clip's start rounded to 0.01 s, but the 9/28 pause times were taken relative to the unrounded
start (the clip the listener heard). That start is recovered from the 9/28 key itself: the offset between the local
absolute pause times and the stored relative ones (the pause counts per clip are identical).
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/pause_check_local.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np

import localpaths as LP
import features as F
import disfl as D
import disfl_io as IO


def pauses(seg, s0: float, s1: float, old: list) -> tuple[list[list[float]], float]:
    found = D.silent_pauses(seg, (s0 - 0.01, s1 + 0.01))
    found = [(a, b) for a, b in found if a >= s0 - 0.006 and b <= s1 + 0.006]
    if old and len(old) == len(found):
        start = float(np.median([a - o[0] for (a, _), o in zip(found, old)]))   # unrounded clip start used on 9/28
    else:
        start = s0
    inside = [(a, b) for a, b in D.silent_pauses(seg, (start, start + 30.0))]
    return [[round(a - start, 3), round(b - start, 3)] for a, b in inside], start


def main() -> None:
    key = json.load(open(LP.PC_KEY_CLOUD, encoding="utf-8"))
    new_key, worst, count_changes = [], 0.0, 0
    vseg_cache: dict[str, np.ndarray] = {}
    for k in key:
        sid, s0, s1 = k["sample_id"], float(k["start_s"]), float(k["end_s"])
        _, seg, _ = IO.rec(sid)
        if sid not in vseg_cache:
            _, x = F.read_pcm(LP.student_wav(sid))
            vseg_cache[sid] = F.speech_segments(F.vad_decisions(x))
        row = dict(k)
        for det, segs in (("praat_pauses", seg), ("webrtc_pauses", vseg_cache[sid])):
            old_list = k[det] if isinstance(k[det], list) else json.loads(k[det])
            row[det], row[det + "_clip_start_unrounded"] = pauses(segs, s0, s1, old_list)
        for det in ("praat_pauses", "webrtc_pauses"):
            old = k[det] if isinstance(k[det], list) else json.loads(k[det])
            if len(old) != len(row[det]):
                count_changes += 1
            else:
                for a, b in zip(old, row[det]):
                    worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
        new_key.append(row)
    LP.PC_KEY.write_text(json.dumps(new_key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"program pauses vs 9/28 key: {count_changes} clip-detector pairs with a different count; "
          f"largest boundary difference {worst:.4f} s")

    res = subprocess.run([sys.executable, str(LP.ROOT / "scripts/local_rerun_20261015/pause_check_analysis.py"),
                          str(LP.PC_LABELS), str(LP.PC_KEY), str(LP.RUN / "features_private.csv")],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(res.stdout)

    # share of program (Praat) pauses >= 1 s among flagged vs other pauses, same clips as the analysis
    rows = list(csv.DictReader(open(LP.PC_LABELS, encoding="utf-8-sig")))
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["clip_id"], []).append(r)
    clips = sorted(c for c, rs in by.items() if float(rs[0]["heard_frac"]) >= 0.95 and rs[0]["confidence"])
    kk = {k["clip_id"]: k for k in new_key}
    flagged, other = [], []
    for c in clips:
        human = [(float(r["start_s"]), float(r["end_s"])) for r in by[c] if r["type"] == "pause"]
        for a, b in kk[c]["praat_pauses"]:
            hit = any(min(b, h1) - max(a, h0) > 0 for h0, h1 in human)
            (flagged if hit else other).append(b - a)
    out["share_at_least_1s"] = {"flagged": [int(sum(d >= 1.0 - 1e-9 for d in flagged)), len(flagged)],
                                "other": [int(sum(d >= 1.0 - 1e-9 for d in other)), len(other)]}
    (LP.OUT / "pause_check_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("praat_pauses", "webrtc_pauses", "share_at_least_1s")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
