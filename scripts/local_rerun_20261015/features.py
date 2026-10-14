"""Whole-recording fluency/prosody features for the 2026-09-26 redo (plan sha e399d945...)."""
from __future__ import annotations
import subprocess, sys, os, json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

import localpaths  # noqa: F401  (adds parselmouth to sys.path)
import parselmouth  # noqa: E402  Praat 6.1.38 inside parselmouth 0.4.7 (cloud: praat_barren 6.4.27)
import _webrtcvad  # noqa: E402  from webrtcvad-wheels 2.0.14 (cloud: py-webrtcvad built from source)

FRAME = 0.030
MIN_SPEECH = 0.09
PAUSE_MIN = 0.25
SILENCE_DB = -25.0
MIN_DIP = 2.0

def read_pcm(path):
    sr, x = wavfile.read(path)
    assert sr == 16000 and x.dtype == np.int16 and x.ndim == 1, (sr, x.dtype, x.shape)
    return sr, x

def vad_decisions(x, sr=16000, mode=2):
    v = _webrtcvad.create(); _webrtcvad.init(v); _webrtcvad.set_mode(v, mode)
    n = int(sr * FRAME); k = len(x) // n
    return np.array([_webrtcvad.process(v, sr, x[i*n:(i+1)*n].tobytes(), n) for i in range(k)], bool)

def speech_segments(dec, frame=FRAME, min_speech=MIN_SPEECH, merge_below=PAUSE_MIN):
    """Speech runs -> drop runs < min_speech -> merge gaps < merge_below. No padding."""
    runs = []; start = None
    for i, d in enumerate(dec):
        if d and start is None: start = i
        elif not d and start is not None: runs.append([start*frame, i*frame]); start = None
    if start is not None: runs.append([start*frame, len(dec)*frame])
    runs = [r for r in runs if r[1]-r[0] >= min_speech - 1e-9]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_below - 1e-9: merged[-1][1] = r[1]
        else: merged.append(list(r))
    return np.array(merged, float).reshape(-1, 2)

def run_praat(wav, prefix):
    # same Praat script, run by the Praat library inside parselmouth instead of the command-line Praat
    parselmouth.praat.run_file(str(Path(__file__).with_name("extract.praat")), str(wav), str(prefix))
    im = np.loadtxt(f"{prefix}_int_meta.txt"); ints = np.loadtxt(f"{prefix}_int.txt").reshape(-1)
    fm = np.loadtxt(f"{prefix}_f0_meta.txt"); f0 = np.loadtxt(f"{prefix}_f0.txt").reshape(-1)
    it = im[0] + im[1]*np.arange(len(ints)); ft = fm[0] + fm[1]*np.arange(len(f0))
    sil = []
    for line in Path(f"{prefix}_sil.txt").read_text().split("\n"):
        if line.strip():
            lab, a, b = line.split(); sil.append((lab, float(a), float(b)))
    return {"int_t": it, "int_db": ints, "f0_t": ft, "f0": f0, "pitch_range": (fm[3], fm[4]), "silences": sil}

def syllable_nuclei(int_t, int_db, f0_t, f0, segments, silence_db=SILENCE_DB, min_dip=MIN_DIP):
    """De Jong & Wempe (2009): intensity peaks above (q99 - 25 dB) whose dip to the next peak exceeds 2 dB,
    voiced at the peak; additionally restricted to VAD speech segments."""
    q99 = np.quantile(int_db, 0.99); thr = max(q99 + silence_db, int_db.min())
    i = np.arange(1, len(int_db)-1)
    peaks = i[(int_db[i] > int_db[i-1]) & (int_db[i] >= int_db[i+1]) & (int_db[i] > thr)]
    valid = []
    for a, b in zip(peaks, list(peaks[1:]) + [None]):
        if b is None:
            valid.append(a); break  # last peak kept if above threshold (as in the script, final peak has no following dip)
        dip = int_db[a:b+1].min()
        if int_db[a] - dip > min_dip: valid.append(a)
    valid = np.array(valid, int)
    t = int_t[valid] if len(valid) else np.array([])
    # voicing at peak: nearest pitch frame within 10 ms voiced
    if len(t):
        j = np.clip(np.searchsorted(f0_t, t), 1, len(f0_t)-1)
        j = np.where(np.abs(f0_t[j-1]-t) < np.abs(f0_t[j]-t), j-1, j)
        t = t[(f0[j] > 0) & (np.abs(f0_t[j]-t) <= 0.011)]
    return t[in_segments(t, segments)]

def in_segments(t, seg):
    t = np.asarray(t, float)
    if len(seg) == 0 or len(t) == 0: return np.zeros(len(t), bool)
    k = np.searchsorted(seg[:, 0], t, side="right") - 1
    ok = k >= 0
    out = np.zeros(len(t), bool); out[ok] = t[ok] < seg[k[ok], 1]
    return out

def clip_segments(seg, a, b):
    s = np.clip(seg, a, b); return s[s[:, 1] - s[:, 0] > 1e-9]

def features_in(interval, seg_all, pr, syl_all):
    a, b = interval
    seg = clip_segments(seg_all, a, b)
    if len(seg) == 0: return None
    phon = float(np.sum(seg[:, 1]-seg[:, 0])); span = float(seg[-1, 1]-seg[0, 0])
    gaps = seg[1:, 0] - seg[:-1, 1]
    pauses = gaps[gaps >= PAUSE_MIN - 1e-9]
    syl = syl_all[(syl_all >= a) & (syl_all < b)]
    nsyl = len(syl)
    # runs between pauses -> syllables per run
    bounds = np.r_[seg[0, 0], [ (seg[i, 1]+seg[i+1, 0])/2 for i in range(len(seg)-1) if gaps[i] >= PAUSE_MIN-1e-9], seg[-1, 1]]
    runs = np.histogram(syl, bins=bounds)[0] if len(bounds) > 1 else np.array([nsyl])
    f0m = (pr["f0"] > 0) & in_segments(pr["f0_t"], seg)
    f0 = pr["f0"][f0m]
    im = in_segments(pr["int_t"], seg)
    idb = pr["int_db"][im]
    st = 12*np.log2(f0/np.median(f0)) if len(f0) >= 20 else np.array([])
    return {
        "articulation_rate": nsyl/phon,
        "pause_rate_per_min": len(pauses)/phon*60,
        "mean_pause_s": float(pauses.mean()) if len(pauses) else np.nan,
        "f0_sd_st": float(np.std(st)) if len(st) else np.nan,
        "intensity_sd_db": float(np.std(idb)) if len(idb) >= 20 else np.nan,
        # extended
        "speech_rate": nsyl/span,
        "phonation_ratio": phon/span,
        "mean_length_of_run": float(runs.mean()),
        "f0_range_st": float(np.subtract(*np.percentile(st, [90, 10]))) if len(st) else np.nan,
        "median_pause_s": float(np.median(pauses)) if len(pauses) else np.nan,
        "long_pause_rate_per_min": float(np.sum(pauses >= 1.0))/phon*60,
        # bookkeeping
        "n_syllables": nsyl, "n_pauses": int(len(pauses)), "phonation_s": phon, "span_s": span,
        "n_voiced_frames": int(len(f0)),
    }

P5 = ["articulation_rate", "pause_rate_per_min", "mean_pause_s", "f0_sd_st", "intensity_sd_db"]
EXT = P5 + ["speech_rate", "phonation_ratio", "mean_length_of_run", "f0_range_st", "median_pause_s", "long_pause_rate_per_min"]

def praat_segments(sil):
    return np.array([[a, b] for lab, a, b in sil if lab == "sounding"], float).reshape(-1, 2)

def analyse(wav, workdir):
    sr, x = read_pcm(wav)
    dur = len(x)/sr
    pr = run_praat(wav, Path(workdir)/Path(wav).stem)
    seg = praat_segments(pr["silences"])                 # primary (deviation 1)
    vseg = speech_segments(vad_decisions(x, sr))          # control detector
    syl = syllable_nuclei(pr["int_t"], pr["int_db"], pr["f0_t"], pr["f0"], seg)
    whole = features_in((0, dur), seg, pr, syl)
    half = dur/2
    h1 = features_in((0, half), seg, pr, syl); h2 = features_in((half, dur), seg, pr, syl)
    cum = np.cumsum(seg[:, 1]-seg[:, 0]); k = np.searchsorted(cum, 48.0)
    end48 = seg[k, 1] - (cum[k]-48.0) if k < len(seg) else dur
    first48 = features_in((0, end48), seg, pr, syl)
    vg = vseg[1:, 0]-vseg[:-1, 1]
    vad_pause_rate = float(np.sum(vg >= PAUSE_MIN-1e-9))/max(float(np.sum(vseg[:, 1]-vseg[:, 0])), 1e-9)*60
    return {"duration_s": dur, "whole": whole, "half1": h1, "half2": h2, "first48": first48, "end48_s": float(end48),
            "vad_pause_rate_per_min": vad_pause_rate, "vad_speech_s": float(np.sum(vseg[:, 1]-vseg[:, 0])),
            "pitch_floor_ceiling": list(pr["pitch_range"]), "praat_speech_s": float(np.sum(seg[:, 1]-seg[:, 0])),
            "n_segments": int(len(seg))}
