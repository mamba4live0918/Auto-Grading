"""Station 6 disfluency measures (plan: docs/research/2026-09-28-disfluency-plan.md, sha aaf0740d...).
D1 filled-pause detector (acoustic), D2 long-pause rate, D3 mid-clause pause rate."""
from __future__ import annotations
import re, json
import numpy as np
from scipy.fft import rfft, dct
import features as F

SR = 16000
WIN, HOP, NFFT, NMEL = 400, 160, 512, 24
GRID = [(d, q, r) for d in (0.20, 0.25, 0.30, 0.40) for q in (15, 25, 35) for r in (1, 2, 3)]

def mel_fb(sr=SR, nfft=NFFT, nmel=NMEL, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    m = lambda f: 2595 * np.log10(1 + f / 700); mi = lambda v: 700 * (10 ** (v / 2595) - 1)
    pts = mi(np.linspace(m(fmin), m(fmax), nmel + 2)); bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((nmel, nfft // 2 + 1))
    for i in range(nmel):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, b): fb[i, k] = (k - a) / max(b - a, 1)
        for k in range(b, c): fb[i, k] = (c - k) / max(c - b, 1)
    return fb

_FB = mel_fb()

def cepstral_change(x):
    """times (frame centres, s) and smoothed Euclidean distance between consecutive c1..c12 vectors."""
    x = x.astype(float)
    n = 1 + max(0, (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx] * np.hamming(WIN)[None, :]
    P = np.abs(rfft(fr, NFFT, axis=1)) ** 2
    E = np.log(P @ _FB.T + 1e-6)
    C = dct(E, type=2, norm='ortho', axis=1)[:, 1:13]
    ch = np.r_[0.0, np.linalg.norm(np.diff(C, axis=0), axis=1)]
    ch = np.convolve(ch, np.ones(5) / 5, mode='same')
    t = (np.arange(n) * HOP + WIN / 2) / SR
    return t, ch

def frame_inputs(x, pr, seg):
    """per-frame change, f0 (Hz, 0 = unvoiced) and in-speech flag on the 10-ms cepstral grid."""
    t, ch = cepstral_change(x)
    f0 = np.interp(t, pr["f0_t"], pr["f0"], left=0, right=0)
    # interpolation across an unvoiced neighbour gives a spurious value -> require nearest pitch frame voiced
    k = np.clip(np.searchsorted(pr["f0_t"], t), 0, len(pr["f0"]) - 1)
    k0 = np.clip(k - 1, 0, len(pr["f0"]) - 1)
    near = np.where(np.abs(pr["f0_t"][k] - t) < np.abs(pr["f0_t"][k0] - t), k, k0)
    f0 = np.where(pr["f0"][near] > 0, f0, 0.0)
    insp = F.in_segments(t, seg)
    return t, ch, f0, insp

def detect(t, ch, f0, insp, d, q, r, interval=None):
    """filled-pause events [(a,b)] inside interval (default whole)."""
    voiced = (f0 > 0) & insp
    if voiced.sum() < 10: return []
    thr = np.percentile(ch[voiced], q)           # per-recording threshold over all voiced frames
    stable = voiced & (ch <= thr)
    if interval is not None:
        stable &= (t >= interval[0]) & (t < interval[1])
    runs, i, n = [], 0, len(t)
    while i < n:
        if stable[i]:
            j = i
            while j + 1 < n and stable[j + 1]: j += 1
            runs.append([i, j]); i = j + 1
        else: i += 1
    merged = []
    for a, b in runs:                             # merge gaps < 50 ms (i.e. <= 4 missing frames)
        if merged and a - merged[-1][1] - 1 < 5: merged[-1][1] = b
        else: merged.append([a, b])
    ev = []
    for a, b in merged:
        dur = (b - a + 1) * HOP / SR
        fv = f0[a:b + 1]; fv = fv[fv > 0]
        if dur < d - 1e-9 or len(fv) < 2: continue
        st = 12 * np.log2(fv / np.median(fv))
        if st.max() - st.min() <= r:
            ev.append((float(t[a] - HOP / SR / 2), float(t[b] + HOP / SR / 2)))
    return ev

def phonation(seg, interval):
    s = F.clip_segments(seg, *interval)
    return float(np.sum(s[:, 1] - s[:, 0])) if len(s) else 0.0

def silent_pauses(seg, interval, min_len=F.PAUSE_MIN):
    """silent gaps between Praat sounding segments fully inside interval (same convention as features_in)."""
    s = F.clip_segments(seg, *interval)
    if len(s) < 2: return []
    g = [(s[i, 1], s[i + 1, 0]) for i in range(len(s) - 1)]
    return [(a, b) for a, b in g if b - a >= min_len - 1e-9]

PUNCT = re.compile(r"[.,?!;:]$")

def classify_pauses(pauses, words, tol=0.15):
    """words: list of (start, end, text). returns list of 'boundary' / 'mid' / None per pause."""
    ws = sorted(words); ends = np.array([w[1] for w in ws]); starts = np.array([w[0] for w in ws])
    out = []
    for a, b in pauses:
        i = np.searchsorted(ends, a + tol, side='right') - 1
        j = np.searchsorted(starts, b - tol, side='left')
        if i < 0 or j >= len(ws): out.append(None); continue
        out.append('boundary' if PUNCT.search(ws[i][2].strip()) else 'mid')
    return out

def words_of(rec):
    if not rec.get('transcription'): return []
    return [(float(w['start']), float(w['end']), w['word']) for s in rec['transcription']['segments'] for w in s.get('words', [])]

def d_measures(seg, fr, words, interval, params):
    t, ch, f0, insp = fr
    ph = phonation(seg, interval) / 60.0
    ev = detect(t, ch, f0, insp, *params, interval=interval)
    ps = silent_pauses(seg, interval)
    cls = classify_pauses(ps, words)
    if ph <= 0: return dict(D1=np.nan, D2=np.nan, D3=np.nan, n_fp=len(ev))
    return dict(D1=len(ev) / ph, D2=sum(1 for a, b in ps if b - a >= 1.0 - 1e-9) / ph,
                D3=sum(1 for c in cls if c == 'mid') / ph, n_fp=len(ev), n_pause=len(ps),
                n_mid=sum(1 for c in cls if c == 'mid'), n_boundary=sum(1 for c in cls if c == 'boundary'))
