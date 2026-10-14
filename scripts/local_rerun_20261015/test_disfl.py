"""Synthetic tests for the D1 filled-pause detector and D3 classifier."""
import numpy as np, tempfile, os
from scipy.io import wavfile
import features as F, disfl as D
SR=16000; rng=np.random.default_rng(0)
def vowel(dur, f0a, f0b, formants_seq):
    n=int(dur*SR); t=np.arange(n)/SR
    f0=np.linspace(f0a,f0b,n); ph=2*np.pi*np.cumsum(f0)/SR
    src=sum(np.sin(k*ph)/k for k in range(1,30))
    out=np.zeros(n); seglen=n//len(formants_seq)
    from scipy.signal import lfilter
    for i,(F1,F2) in enumerate(formants_seq):
        a=i*seglen; b=n if i==len(formants_seq)-1 else (i+1)*seglen
        y=src[a:b]
        for Fc,bw in ((F1,80),(F2,120)):
            r=np.exp(-np.pi*bw/SR); th=2*np.pi*Fc/SR
            y=lfilter([1-r],[1,-2*r*np.cos(th),r*r],y)
        out[a:b]=y
    return out/np.max(np.abs(out))*0.5
def sil(d): return rng.normal(0,1e-4,int(d*SR))
speechy=lambda d: vowel(d,110,190,[(rng.uniform(300,800),rng.uniform(900,2400)) for _ in range(int(d/0.07))])
sig=np.concatenate([sil(0.5),speechy(2.0),sil(0.4),vowel(0.6,140,141,[(600,1000)]),sil(0.4),speechy(2.0),sil(0.5)])
fp=(0.5+2.0+0.4, 0.5+2.0+0.4+0.6)
with tempfile.TemporaryDirectory() as td:
    w=os.path.join(td,'syn.wav'); wavfile.write(w,SR,(sig*32767).astype(np.int16))
    pr=F.run_praat(w, os.path.join(td,'syn')); seg=F.praat_segments(pr['silences'])
    sr,x=F.read_pcm(w); fr=D.frame_inputs(x,pr,seg)
    ev=D.detect(*fr,0.25,25,2)
    print('events',[(round(a,2),round(b,2)) for a,b in ev],'target',fp)
    hit=[e for e in ev if min(e[1],fp[1])-max(e[0],fp[0])>0.2]
    fa=[e for e in ev if min(e[1],fp[1])-max(e[0],fp[0])<=0]
    assert len(hit)==1, 'steady vowel not detected'
    assert len(fa)==0, 'false alarm in gliding speech'
# D3 classifier
words=[(0,0.3,'Hello,'),(0.3,0.6,'my'),(0.6,0.9,'name'),(1.5,1.8,'is'),(1.8,2.0,'Bob.'),(2.6,2.9,'I')]
cls=D.classify_pauses([(0.9,1.5),(2.0,2.6),(3.0,3.5)],words)
assert cls==['mid','boundary',None], cls
print('all tests passed')
