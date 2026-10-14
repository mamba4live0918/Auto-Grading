import numpy as np, litcues as L
# synthetic praat dict: rising end vs falling end
t = np.arange(0, 10, 0.01); f0 = np.full_like(t, 120.0)
m_ = (t > 2.7) & (t <= 3.0); f0[m_] = np.linspace(120, 140, m_.sum())   # rise at word 1 end (3.0)
m_ = (t > 5.7) & (t <= 6.0); f0[m_] = np.linspace(120, 100, m_.sum())   # fall at word 2 end
m_ = (t > 8.7) & (t <= 9.0); f0[m_] = np.linspace(120, 145, m_.sum())   # rise
it = np.arange(0, 10, 0.016); idb = np.full_like(it, 70.0); idb[(it >= 5.5) & (it <= 6.0)] = 60
pr = {"f0_t": t, "f0": f0, "int_t": it, "int_db": idb}
words = [(2.5, 3.0, "done."), (5.5, 6.0, "fine."), (8.5, 9.0, "ok."), (9.1, 9.5, "why?")]
seg = np.array([[0, 10.0]])
p1 = L.P1_final_rise(pr, words, (0, 10)); assert abs(p1 - 2/3) < 1e-9, p1
l2 = L.L2_end_loudness(pr, seg, words, (0, 10)); assert abs(l2 - 0.0) < 1e-9, l2   # median of [0,-10,0]
print("litcues tests passed", p1, l2)
