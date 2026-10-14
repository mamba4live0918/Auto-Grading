import numpy as np, tempfile
from pathlib import Path
from scipy.io import wavfile
import features as F
SR=16000
rng=np.random.default_rng(0)
def voiced(dur,f0=150,amp=0.3,am=4.0):
    t=np.arange(int(dur*SR))/SR
    x=sum(np.sin(2*np.pi*f0*k*t)/k for k in range(1,8))
    env=0.55+0.45*np.sin(2*np.pi*am*t-np.pi/2)
    return amp*x*env
def silence(d): return rng.normal(0,1e-4,int(d*SR))
def write(x,path):
    wavfile.write(path,SR,np.clip(x*32767/np.max(np.abs(x))*0.8,-32768,32767).astype(np.int16))

def test_pause_detection():
    parts=[silence(0.5)];gaps=[0.10,0.30,0.60,1.20]
    for g in gaps: parts+= [voiced(2.0),silence(g)]
    parts+=[voiced(2.0),silence(0.5)]
    x=np.concatenate(parts)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'t.wav';write(x,p);r=F.analyse(p,d)
    w=r['whole'];print('pauses',w['n_pauses'],'mean',w['mean_pause_s'],'phon',w['phonation_s'],'vad rate',r['vad_pause_rate_per_min'])
    assert w['n_pauses']==3, w['n_pauses']
    assert abs(w['mean_pause_s']-np.mean([0.3,0.6,1.2]))<0.08
    assert abs(w['phonation_s']-(10+0.10))<0.4   # 0.1 s gap merged into speech

def test_syllable_count():
    # 20 syllables: 160 ms voiced bursts, 90 ms low-level valleys, inside one speech stretch
    parts=[silence(0.5)]
    for i in range(20): parts+=[voiced(0.16,am=1/0.32),0.02*voiced(0.09)]
    parts+=[silence(0.5)]
    x=np.concatenate(parts)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'t.wav';write(x,p);r=F.analyse(p,d)
    n=r['whole']['n_syllables'];print('syllables',n)
    assert 18<=n<=22,n

def test_pitch_and_intensity():
    t=np.arange(int(3*SR))/SR
    glide=100*2**(t/3)  # 100->200 Hz, 12 st
    ph=2*np.pi*np.cumsum(glide)/SR
    x=0.3*sum(np.sin(k*ph)/k for k in range(1,6))
    x=np.concatenate([silence(0.5),x,silence(0.5)])
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'t.wav';write(x,p);r=F.analyse(p,d)
    w=r['whole'];print('f0 range st',w['f0_range_st'],'sd',w['f0_sd_st'])
    assert 8.5<w['f0_range_st']<10.5   # p10-p90 of a uniform 12-st glide = 9.6
    assert 3.0<w['f0_sd_st']<3.9       # sd of uniform over 12 st = 3.46
    # constant level tone: intensity sd small
    y=np.concatenate([silence(0.5),0.3*np.sin(2*np.pi*200*np.arange(3*SR)/SR),silence(0.5)])
    # alternating 6 dB steps every 0.5 s
    z=0.3*np.sin(2*np.pi*200*np.arange(4*SR)/SR)*np.repeat([1,0.5]*4,SR//2)
    z=np.concatenate([silence(0.5),z,silence(0.5)])
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'y.wav';write(y,p);ry=F.analyse(p,d)
        q=Path(d)/'z.wav';write(z,q);rz=F.analyse(q,d)
    print('int sd const',ry['whole']['intensity_sd_db'],'alt6dB',rz['whole']['intensity_sd_db'])
    assert ry['whole']['intensity_sd_db']<0.5
    assert 2.4<rz['whole']['intensity_sd_db']<3.6   # sd of +-3 dB square wave = 3

def test_segment_logic():
    dec=np.array([0]*5+[1]*10+[0]*7+[1]*2+[0]*3+[1]*10+[0]*5,bool)  # 2-frame blip (60ms) removed; gap 7 frames=210ms merged? no: 0.21<0.25 -> merged
    s=F.speech_segments(dec)
    assert len(s)==2 and abs((s[1,0]-s[0,1])-0.36)<1e-9   # 60 ms blip dropped, so 210+60+90 ms = one 360 ms pause
    dec1=np.array([0]*5+[1]*10+[0]*7+[1]*10,bool)          # 210 ms gap < 250 ms -> merged
    assert len(F.speech_segments(dec1))==1
    dec2=np.array([0]*5+[1]*10+[0]*9+[1]*10,bool)  # 270 ms gap -> pause
    s2=F.speech_segments(dec2);assert len(s2)==2 and abs((s2[1,0]-s2[0,1])-0.27)<1e-9

if __name__=='__main__':
    for f in [test_segment_logic,test_pause_detection,test_syllable_count,test_pitch_and_intensity]:
        f();print('PASS',f.__name__)
