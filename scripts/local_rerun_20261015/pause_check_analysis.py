"""Station 2 analysis (plan: docs/research/2026-09-28-pause-check-plan.md, deviations 1-4).
Final design: the listener hears each whole 30-s clip (replays allowed) and drags ranges ONLY for
problem events: 'pause' = disfluent / hesitation pause (normal boundary pauses are NOT marked),
'uh' = filled pause, 'repeat' = repetition / self-repair. Plus one 1-5 confidence rating per clip.
Usage: python3 pause_check_analysis.py <卡顿标注.csv> <pause_check_key_DO_NOT_OPEN.json> [features_private.csv]
Needs only numpy. All results are descriptive (no p-values)."""
import sys, csv, json, numpy as np

def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v; r[m] = r[m].mean()
    return r
def spearman(x, y):
    if len(x) < 3: return None
    x, y = rank(x), rank(y)
    if x.std() == 0 or y.std() == 0: return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)
def auroc(pos, neg):
    if not pos or not neg: return None
    r = rank(list(pos) + list(neg)); n1 = len(pos)
    return round(float((r[:n1].sum() - n1 * (n1 - 1) / 2) / (n1 * len(neg))), 3)
def med(v, d=3): return round(float(np.median(v)), d) if len(v) else None

rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')))
key = {k['clip_id']: k for k in json.load(open(sys.argv[2]))}
def prog(c, det):
    v = key[c][det]; return [tuple(map(float, p)) for p in (json.loads(v) if isinstance(v, str) else v)]
by = {}
for r in rows: by.setdefault(r['clip_id'], []).append(r)
clips = sorted(c for c, rs in by.items() if float(rs[0]['heard_frac']) >= 0.95 and rs[0]['confidence'])
H = {c: {t: [(float(r['start_s']), float(r['end_s'])) for r in by[c] if r['type'] == t] for t in ['pause', 'uh', 'repeat']} for c in clips}
ov = lambda x, y: min(x[1], y[1]) - max(x[0], y[0])
out = {'clips_done': len(clips),
       'human': {t: {'total': sum(len(H[c][t]) for c in clips),
                     'mean_per_clip': round(float(np.mean([len(H[c][t]) for c in clips])), 2) if clips else None,
                     'clips_with_any': sum(1 for c in clips if H[c][t]),
                     'median_duration_s': med([b - a for c in clips for a, b in H[c][t]])} for t in ['pause', 'uh', 'repeat']}}
for det in ['praat_pauses', 'webrtc_pauses']:
    found, flagged_d, other_d, n_prog, n_flag = 0, [], [], 0, 0
    per_clip_prog, per_clip_human = [], []
    for c in clips:
        P, HP = prog(c, det), H[c]['pause']
        found += sum(1 for h in HP if any(ov(h, p) > 0 for p in P))
        for p in P:
            n_prog += 1
            if any(ov(h, p) > 0 for h in HP): flagged_d.append(p[1] - p[0]); n_flag += 1
            else: other_d.append(p[1] - p[0])
        per_clip_prog.append(len(P)); per_clip_human.append(len(HP))
    nH = sum(len(H[c]['pause']) for c in clips)
    out[det] = {
        'human_problem_pauses_that_program_also_found': f'{found}/{nH}',
        'program_pauses_marked_as_problem_by_human': f'{n_flag}/{n_prog}',
        'median_duration_s_problem_vs_other_program_pauses': [med(flagged_d), med(other_d)],
        'duration_auroc_problem_vs_other': auroc(flagged_d, other_d),
        'per_clip_spearman_program_pause_count_vs_human_problem_pauses': spearman(per_clip_prog, per_clip_human),
    }
conf = [int(by[c][0]['confidence']) for c in clips]
out['confidence_distribution'] = {str(v): conf.count(v) for v in range(1, 6)}
tot = lambda c: sum(len(H[c][t]) for t in H[c])
out['confidence_spearman_with_human_counts'] = {t: spearman(conf, [len(H[c][t]) for c in clips]) for t in ['pause', 'uh', 'repeat']}
out['confidence_spearman_with_human_counts']['all_three'] = spearman(conf, [tot(c) for c in clips])
if len(sys.argv) > 3:
    F = {r['sample_id']: r for r in csv.DictReader(open(sys.argv[3], encoding='utf-8-sig'))}
    P5 = ['articulation_rate', 'pause_rate_per_min', 'mean_pause_s', 'f0_sd_st', 'intensity_sd_db']
    out['confidence_spearman_with_whole_recording_P5'] = {f: spearman(conf, [float(F[key[c]['sample_id']]['whole__' + f]) for c in clips]) for f in P5}
print(json.dumps(out, ensure_ascii=False, indent=1))
