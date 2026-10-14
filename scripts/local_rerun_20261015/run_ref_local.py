"""Professional speakers (Europarl-ASR, 500 selected speeches): features.analyse on the first 180 s (main) and on
the full speech (robustness), locally. Same audio as 9/26 (the selected 16 kHz mono PCM16 speeches); the 180-s
version keeps the first 180 x 16000 samples.
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/run_ref_local.py [main|full]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import localpaths as LP

PRAAT_TMP = Path("D:/ThesisLocalRerun_20261015/ref_praat")
SECONDS = 180


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer): return int(v)
    return v


def cut_180(src: Path) -> Path:
    dest = LP.REF180_WAV / src.name
    if not dest.exists():
        sr, x = wavfile.read(src)
        assert sr == 16000 and x.dtype == np.int16 and x.ndim == 1
        wavfile.write(dest, sr, x[: SECONDS * sr])
    return dest


def one(args: tuple[str, str]) -> tuple[str, str]:
    import features as F
    mode, sid = args
    out_dir = LP.REF_OUT if mode == "main" else LP.REF_OUT_FULL
    dest = out_dir / f"{sid}.json"
    if dest.exists():
        return sid, "cached"
    src = LP.REF_WAV_DIR / f"{sid}.wav"
    wav = cut_180(src) if mode == "main" else src
    work = PRAAT_TMP / mode
    try:
        r = F.analyse(wav, work)
    except Exception as e:  # same convention as 9/26: keep going, record the error
        dest.write_text(json.dumps({"error": repr(e)}))
        return sid, "ERROR " + repr(e)
    dest.write_text(json.dumps(clean(r)))
    return sid, "done"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    assert mode in ("main", "full")
    LP.REF180_WAV.mkdir(parents=True, exist_ok=True)
    (PRAAT_TMP / mode).mkdir(parents=True, exist_ok=True)
    clips = json.load(open(LP.REF_MANIFEST, encoding="utf-8"))["clips"]
    sids = sorted(c["speaker_hash"] for c in clips)
    assert len(sids) == 500 and all((LP.REF_WAV_DIR / f"{s}.wav").exists() for s in sids)
    t = time.time()
    with ProcessPoolExecutor(10) as ex:
        for i, (sid, st) in enumerate(ex.map(one, [(mode, s) for s in sids])):
            if i % 50 == 0 or st.startswith("ERROR"):
                print(mode, i, sid[:8], st, round(time.time() - t), flush=True)
    print("ALL DONE", mode, round(time.time() - t), flush=True)
