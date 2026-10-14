import json, numpy as np
from pathlib import Path
import features as F, disfl as D
import localpaths as LP
PR = LP.PRAAT_DIR
TR = str(LP.TR_A)
def load_praat(sid):
    p = PR / sid
    im = np.loadtxt(f"{p}_int_meta.txt"); ints = np.loadtxt(f"{p}_int.txt").reshape(-1)
    fm = np.loadtxt(f"{p}_f0_meta.txt"); f0 = np.loadtxt(f"{p}_f0.txt").reshape(-1)
    sil = [(l.split()[0], float(l.split()[1]), float(l.split()[2])) for l in open(f"{p}_sil.txt").read().split("\n") if l.strip()]
    return {"int_t": im[0] + im[1] * np.arange(len(ints)), "int_db": ints, "f0_t": fm[0] + fm[1] * np.arange(len(f0)), "f0": f0, "silences": sil}
_cache = {}
def rec(sid):
    if sid not in _cache:
        pr = load_praat(sid); seg = F.praat_segments(pr['silences'])
        sr, x = F.read_pcm(LP.student_wav(sid))
        _cache[sid] = (pr, seg, D.frame_inputs(x, pr, seg))
    return _cache[sid]
_W = None
def words(sid):
    global _W
    if _W is None:
        _W = {}
        for l in open(TR, encoding='utf-8'):
            r = json.loads(l); _W[r['sample_id']] = D.words_of(r)
    return _W[sid]
