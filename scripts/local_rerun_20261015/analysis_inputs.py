import csv,json,numpy as np, features as F
import localpaths as LP
MAN=str(LP.MAN)
rows=[r for r in csv.DictReader(open(MAN,encoding='utf-8')) if r['score_status']=='observed' and r['audio_status']=='succeeded' and r['feature_status']=='succeeded' and r['supervised_eligible']=='True']
ids=sorted(r['sample_id'] for r in rows)
rec={i:json.load(open(LP.OUT/f'{i}.json')) for i in ids}
Xp=np.array([[np.nan if rec[i]['whole'][k] is None else rec[i]['whole'][k] for k in F.P5] for i in ids],float)
