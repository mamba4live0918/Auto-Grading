form args
  sentence infile
  sentence outprefix
endform
s = Read from file: infile$
# ---- intensity (De Jong & Wempe: min pitch 50 Hz, subtract mean) ----
selectObject: s
int = To Intensity: 50, 0, "yes"
nf = Get number of frames
x1 = Get time from frame number: 1
dx = Get time step
m = Down to Matrix
Save as headerless spreadsheet file: outprefix$ + "_int.txt"
writeFileLine: outprefix$ + "_int_meta.txt", x1, " ", dx, " ", nf
# ---- pitch, two-pass (Hirst): pass1 75-600, pass2 q15*0.83 .. q65*1.92 ----
selectObject: s
p1 = To Pitch: 0.01, 75, 600
q15 = Get quantile: 0, 0, 0.15, "Hertz"
q65 = Get quantile: 0, 0, 0.65, "Hertz"
if q15 = undefined
  fl = 75
  ce = 600
else
  fl = q15 * 0.83
  ce = q65 * 1.92
endif
if fl < 40
  fl = 40
endif
selectObject: s
p2 = To Pitch: 0.01, fl, ce
pnf = Get number of frames
px1 = Get time from frame number: 1
pdx = Get time step
pm = To Matrix
Save as headerless spreadsheet file: outprefix$ + "_f0.txt"
writeFileLine: outprefix$ + "_f0_meta.txt", px1, " ", pdx, " ", pnf, " ", fl, " ", ce
# ---- primary pause detector: Praat silences (De Jong & Wempe threshold relative to q99, min silent 0.25, min sounding 0.05) ----
selectObject: s
i100 = To Intensity: 100, 0, "yes"
imax = Get maximum: 0, 0, "Parabolic"
iq99 = Get quantile: 0, 0, 0.99
thr = -25 - (imax - iq99)
writeFileLine: outprefix$ + "_sil_meta.txt", thr
selectObject: s
tg = To TextGrid (silences): 100, 0, thr, 0.25, 0.05, "silent", "sounding"
ni = Get number of intervals: 1
out$ = ""
for i to ni
  lab$ = Get label of interval: 1, i
  a = Get start time of interval: 1, i
  b = Get end time of interval: 1, i
  out$ = out$ + lab$ + " " + fixed$(a, 4) + " " + fixed$(b, 4) + newline$
endfor
writeFile: outprefix$ + "_sil.txt", out$
