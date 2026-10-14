"""Students: per-recording features (features.analyse) and eGeMAPSv02 functionals, locally.

Same inputs as 9/26 (the 82 analysis.wav files, byte-identical); Praat runs inside parselmouth (6.1.38) and
openSMILE through its Python package (opensmile 2.6.0, openSMILE 3.0.1) instead of the command-line tools.
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/run_extract_local.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import localpaths as LP

WAV_LINKS = LP.RUN / "wav"   # hard links named <sid>.wav (Praat output files are named after the WAV stem)


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer): return int(v)
    return v


def analysis_ids() -> list[str]:
    with open(LP.MAN, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["score_status"] == "observed" and r["audio_status"] == "succeeded"
                and r["feature_status"] == "succeeded" and r["supervised_eligible"] == "True"]
    ids = sorted(r["sample_id"] for r in rows)
    assert len(ids) == 82
    return ids


def link(sid: str) -> Path:
    dest = WAV_LINKS / f"{sid}.wav"
    if not dest.exists():
        os.link(LP.student_wav(sid), dest)
    return dest


def egemaps_csv(wav: Path, sid: str, dest: Path) -> None:
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    df = smile.process_file(str(wav))
    assert df.shape == (1, 88), df.shape
    names = list(df.columns)
    vals = df.iloc[0].to_numpy(float)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(";".join(["name", "frameTime"] + names) + "\n")
        f.write(";".join([f"'{sid}'", "0.000000"] + [repr(float(v)) for v in vals]) + "\n")


def one(sid: str) -> tuple[str, str]:
    import features as F
    dest = LP.OUT / f"{sid}.json"
    if dest.exists():
        return sid, "cached"
    wav = link(sid)
    r = F.analyse(wav, LP.PRAAT_DIR)
    r["sha256"] = hashlib.sha256(wav.read_bytes()).hexdigest()
    egemaps_csv(wav, sid, LP.EGEMAPS_DIR / f"{sid}.csv")
    dest.write_text(json.dumps(clean(r)))
    return sid, "done"


if __name__ == "__main__":
    WAV_LINKS.mkdir(parents=True, exist_ok=True)
    ids = analysis_ids()
    t = time.time()
    with ProcessPoolExecutor(8) as ex:
        for sid, st in ex.map(one, ids):
            print(sid, st, round(time.time() - t, 1), flush=True)
