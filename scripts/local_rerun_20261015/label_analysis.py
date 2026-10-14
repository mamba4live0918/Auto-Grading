"""Station 5 analysis after the listening labels arrive (plan: 2026-09-26-reference-and-grouping-plan.md).
Usage: python3 label_analysis.py <听辨结果.csv> <listening_key_DO_NOT_OPEN.csv> <students_ref.npz>"""
import sys,csv,json,numpy as np
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.metrics import roc_auc_score, cohen_kappa_score
lab={r['clip_id']:r['judgement'] for r in csv.DictReader(open(sys.argv[1],encoding='utf-8-sig'))}
key=list(csv.DictReader(open(sys.argv[2],encoding='utf-8')))
d=np.load(sys.argv[3]); ids=list(d['ids']); plike=d['plike']; group=d['group']
out={}
# self-consistency on repeated clips
first={};pairs=[]
for r in key:
    j=lab.get(r['clip_id'],'')
    if not j: continue
    k=(r['sample_id'],r['start_s'])
    if k in first: pairs.append((first[k],j))
    else: first[k]=j
out['repeat_pairs']=len(pairs)
if pairs: out['repeat_agreement']=float(np.mean([a==b for a,b in pairs])); out['repeat_kappa']=float(cohen_kappa_score(*zip(*pairs)))
# one label per recording (first presentation)
per={}
for r in key:
    if r['repeat_of']=='' and lab.get(r['clip_id']): per[r['sample_id']]=lab[r['clip_id']]
out['counts']={v:sum(1 for x in per.values() if x==v) for v in ['好','说不准','不好']}
gb=[(plike[ids.index(s)],group[ids.index(s)],1 if v=='好' else 0) for s,v in per.items() if v in ('好','不好')]
if len(set(x[2] for x in gb))==2:
    p=np.array([x[0] for x in gb]);yl=np.array([x[2] for x in gb]);g=np.array([x[1] for x in gb])
    auc=roc_auc_score(yl,p);rng=np.random.default_rng(20260926)
    perm=np.array([roc_auc_score(rng.permutation(yl),p) for _ in range(999)])
    out['plike_auroc_good_vs_bad']=float(auc); out['perm_p_one_sided']=float((1+np.sum(perm>=auc))/1000)
    tab=[[int(np.sum((g==c)&(yl==1))),int(np.sum((g==c)&(yl==0)))] for c in sorted(set(group))]
    out['group_table_good_bad']=tab
    out['fisher_p']=float(fisher_exact(tab)[1]) if len(tab)==2 else float(chi2_contingency(tab)[1])
print(json.dumps(out,indent=1,ensure_ascii=False))
