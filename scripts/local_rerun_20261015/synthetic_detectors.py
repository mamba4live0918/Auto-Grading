"""The detector comparison behind deviation 1 of the 9/26 plan, recomputed locally: on the synthetic pause signal of
test_features.test_pause_detection (2-s voiced stretches separated by 0.10, 0.30, 0.60 and 1.20 s of near-silence),
how long are the pauses according to Praat, and how far past each true speech end does WebRTC VAD keep "speech"?
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/synthetic_detectors.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import localpaths as LP
import features as F
import test_features as T


def main() -> None:
    T.rng = np.random.default_rng(0)          # same random stream as the test (it is the first user of rng)
    gaps = [0.10, 0.30, 0.60, 1.20]
    parts, t, true_ends = [T.silence(0.5)], 0.5, []
    for g in gaps:
        parts += [T.voiced(2.0), T.silence(g)]
        true_ends.append(t + 2.0)
        t += 2.0 + g
    parts += [T.voiced(2.0), T.silence(0.5)]
    true_ends.append(t + 2.0)
    x = np.concatenate(parts)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.wav"
        T.write(x, p)
        pr = F.run_praat(p, Path(d) / "t")
        _, pcm = F.read_pcm(p)
    seg = F.praat_segments(pr["silences"])
    praat_pauses = [round(float(b - a), 3) for a, b in zip(seg[:-1, 1], seg[1:, 0])]
    vseg = F.speech_segments(F.vad_decisions(pcm))
    vad_pauses = [round(float(b - a), 3) for a, b in zip(vseg[:-1, 1], vseg[1:, 0])]
    # overhang past each true speech end; where WebRTC merged the following silence away there is no segment end
    # near the true end, so that end is reported as merged instead of as a (meaningless) multi-second overhang
    overhang_ms, merged_after_gap_s = [], []
    for e, g in zip(true_ends, gaps + [None]):
        k = int(np.argmin(np.abs(vseg[:, 1] - e)))
        if abs(vseg[k, 1] - e) <= 0.5:
            overhang_ms.append(round(float((vseg[k, 1] - e) * 1000)))
        else:
            merged_after_gap_s.append(g)
    out = {"true_gaps_s": gaps, "praat_pause_durations_s": praat_pauses,
           "webrtc_pause_durations_s": vad_pauses, "webrtc_speech_end_overhang_ms": overhang_ms,
           "webrtc_merged_silences_s": merged_after_gap_s}
    (LP.PUBLIC / "synthetic_detectors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
