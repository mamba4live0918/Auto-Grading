"""Reproduce the binary-50 and score-correlation report without fitting.

Public output is aggregate only. Human labels come from the exported
snapshot; teacher totals come from the original frozen eligible manifest and
are cross-checked against two existing sources. Unit = one recording: 60 human
audit records or 82 teacher-score records, never 730 duplicated teacher labels.
All p values are raw, two-sided, exploratory. Bootstrap intervals describe fixed
predictions resampled by recording; they do not establish population performance.
"""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr,rankdata,spearmanr

ROOT=Path(__file__).resolve().parents[2]
import localpaths as LP
PRIVATE=LP.RUN/'report_correlations'
PUBLIC=LP.PUBLIC/'report_correlations'
HUMAN=Path('C:/Users/Mamba4live/Downloads/30秒分段_人工答案_20260930.json')
FROZEN=ROOT/'data/private/segment_pregrade_20260930/pregrades_initial_ternary_20260930.json'
LIVE=ROOT/'data/private/segment_pregrade_20260930/pregrades.json'
MANIFEST=ROOT/'artifacts/research/444eb952b40469ad9e3dfe8983ea7461335f89f23ca0ed99de465d20d55c10ca/manifest.csv'
CROSSCHECKS=[ROOT/'artifacts/research/a4a7e64ba59438f3cc8dbfb1174e4b6a680db93c4bc6ae4e97925c8e814ca4fe/manifest.csv',ROOT/'data/private/local_source_manifest.csv']
RIDGE=LP.OUT/'preds.npz'
RESAMPLES=10000
SEED=20260930


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def _vectors(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=1 or y.ndim!=1 or len(x)!=len(y) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('paired finite one-dimensional observations required; no silent deletion')
    return x,y


def _nullable(value):return float(value) if np.isfinite(value) else None


def _correlation(x,y,kind):
    if np.ptp(x)==0 or np.ptp(y)==0:
        return {'coefficient':None,'p_raw':None,'status':'undefined_constant_input'}
    result=(spearmanr if kind=='spearman' else pearsonr)(x,y)
    return {'coefficient':_nullable(result.statistic),'p_raw':_nullable(result.pvalue),'status':'available'}


def continuous_correlations(x,y):
    x,y=_vectors(x,y)
    output={'n':len(x),'spearman':_correlation(x,y,'spearman'),'pearson':_correlation(x,y,'pearson')}
    deltas={}
    for name,xx,yy in [('spearman',rankdata(x),rankdata(y)),('pearson',x,y)]:
        value=output[name]['coefficient']
        deltas[name]=abs(value-float(np.corrcoef(xx,yy)[0,1])) if value is not None else None
    output['verification']={'independent_correlation_max_differences':deltas}
    return output


def binary_metrics(scores,labels):
    x,y=_vectors(scores,labels)
    if set(np.unique(y))!={0.,1.}:raise ValueError('both binary human classes are required')
    pred=x>=50;positive=y==1;negative=~positive
    tp=int(np.sum(pred&positive));fp=int(np.sum(pred&negative));tn=int(np.sum(~pred&negative));fn=int(np.sum(~pred&positive))
    n1,n0=int(positive.sum()),int(negative.sum())
    ranks=rankdata(x,method='average');auc=float((ranks[positive].sum()-n1*(n1+1)/2)/(n1*n0))
    differences=x[positive,None]-x[None,negative]
    pairwise=float((np.sum(differences>0)+.5*np.sum(differences==0))/(n1*n0))
    c=continuous_correlations(x,y)
    return {'n':len(x),'threshold':50,'good_rule':'score >= 50','bad_rule':'score < 50',
            'human_good':n1,'human_bad':n0,'machine_good':int(pred.sum()),'machine_bad':int((~pred).sum()),
            'confusion':{'tp':tp,'fp':fp,'tn':tn,'fn':fn},'correct':tp+tn,
            'accuracy':(tp+tn)/len(x),'balanced_accuracy':.5*(tp/n1+tn/n0),
            'sensitivity_good':tp/n1,'specificity_bad':tn/n0,'auroc':auc,
            'point_biserial_pearson':c['pearson'],'spearman':c['spearman'],
            'verification':{'auc_pairwise':pairwise,'auc_rank_pair_difference':abs(auc-pairwise),**c['verification']}}


def _row_corr(x,y):
    a=x-x.mean(axis=1,keepdims=True);b=y-y.mean(axis=1,keepdims=True)
    denominator=np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
    return np.divide(np.sum(a*b,axis=1),denominator,out=np.full(len(x),np.nan),where=denominator>0)


def bootstrap_intervals(x,y,*,kind,resamples=RESAMPLES,seed=SEED):
    x,y=_vectors(x,y)
    if kind not in {'binary','continuous'} or resamples<=0:raise ValueError('invalid bootstrap configuration')
    draws=np.random.default_rng(seed).integers(0,len(x),(resamples,len(x)))
    xx,yy=x[draws],y[draws];rx,ry=rankdata(xx,axis=1),rankdata(yy,axis=1)
    estimates={'pearson':_row_corr(xx,yy),'spearman':_row_corr(rx,ry)}
    if kind=='binary':
        estimates['point_biserial_pearson']=estimates.pop('pearson')
        n1=yy.sum(axis=1);n0=len(x)-n1;denominator=n1*n0
        estimates['auroc']=np.divide((rx*yy).sum(axis=1)-n1*(n1+1)/2,denominator,out=np.full(resamples,np.nan),where=denominator>0)
        pred=xx>=50
        estimates['accuracy']=(pred==(yy==1)).mean(axis=1)
        tp=np.sum(pred&(yy==1),axis=1);tn=np.sum(~pred&(yy==0),axis=1)
        estimates['balanced_accuracy']=.5*(np.divide(tp,n1,out=np.full(resamples,np.nan),where=n1>0)+np.divide(tn,n0,out=np.full(resamples,np.nan),where=n0>0))
    intervals={};valid={}
    for key,values in estimates.items():
        finite=values[np.isfinite(values)];valid[key]=len(finite)
        intervals[key]=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
    return {'method':'paired percentile bootstrap of fixed predictions, descriptive only','resamples':resamples,'seed':seed,
            'resampling_unit':'recording','interval_level':.95,'valid_resamples':valid,'intervals':intervals,
            'models_refitted_in_bootstrap':False}


def load_teacher_scores(path,target_ids,*,expected_count=82,require_eligible=True):
    target_ids=list(target_ids)
    if len(target_ids)!=expected_count or len(set(target_ids))!=expected_count:raise ValueError('teacher target cohort invalid')
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate teacher identity')
    eligible=[r for r in rows if r.get('score_status')=='observed' and
              (not require_eligible or all(r.get(k)==v for k,v in [('audio_status','succeeded'),('feature_status','succeeded'),('supervised_eligible','True')]))]
    mapping={r['sample_id']:float(r['total_score']) for r in eligible}
    if any(sid not in mapping for sid in target_ids):raise ValueError('teacher score missing for target recording')
    if require_eligible and set(mapping)!=set(target_ids):raise ValueError('eligible manifest cohort differs from score cohort')
    values=np.array([mapping[sid] for sid in target_ids],float)
    if not np.isfinite(values).all() or np.any((values<0)|(values>100)):raise ValueError('invalid teacher total score')
    return values,{'manifest_rows':len(rows),'eligible_or_observed_rows':len(eligible),'aligned_recordings':len(values),'unique_recordings':len(set(target_ids))}


def ridge_oof_report(path,target_ids,teacher,*,expected_repeats=10,bootstrap_resamples=RESAMPLES):
    path=Path(path)
    if not path.exists():return {'available':False,'reason':'saved_oof_artifact_missing'},None
    with np.load(path,allow_pickle=False) as data:
        if not {'ids','y','Ridge_P5'}.issubset(data.files):return {'available':False,'reason':'required_saved_oof_arrays_missing'},None
        ids=data['ids'].tolist();observed=data['y'];predictions=data['Ridge_P5']
    if len(ids)!=len(set(ids)) or set(ids)!=set(target_ids):raise ValueError('Ridge OOF recording cohort mismatch')
    if predictions.shape!=(expected_repeats,len(ids)) or not np.isfinite(predictions).all():raise ValueError('Ridge OOF prediction shape or values invalid')
    positions={sid:i for i,sid in enumerate(ids)};order=[positions[sid] for sid in target_ids]
    if not np.array_equal(observed[order],np.asarray(teacher,float)):raise ValueError('Ridge saved target differs from teacher manifest')
    prediction=predictions[:,order];averaged=prediction.mean(axis=0)
    repeatwise=np.array([spearmanr(row,teacher).statistic for row in prediction])
    result={'available':True,'n':len(target_ids),'repeats':expected_repeats,'prediction_kind':'saved outer-fold held-out predictions; no refitting',
            'repeatwise_spearman':{'mean':float(np.mean(repeatwise)),'min':float(np.min(repeatwise)),'max':float(np.max(repeatwise))},
            'mean_oof_prediction_correlations':continuous_correlations(averaged,teacher),
            'bootstrap95':bootstrap_intervals(averaged,teacher,kind='continuous',resamples=bootstrap_resamples),
            'source_provenance':{'generator':'scripts/acoustic_redo/analysis.py:58-78','outer_test_assignment':'scripts/acoustic_redo/model.py:35-45',
                                 'nested_training':'scripts/acoustic_redo/model.py:20-28'},
            'aggregation_note':'Mean of 10 repeatwise correlations is different from correlation of the mean OOF prediction.'}
    return result,averaged


def main():
    if not FROZEN.exists():raise ValueError('frozen pregrade archive required; do not silently switch score sources')
    paths=[HUMAN,FROZEN,LIVE,MANIFEST,*CROSSCHECKS]
    if RIDGE.exists():paths.append(RIDGE)
    hashes={str(path):sha(path) for path in paths}
    frozen,live,human=read(FROZEN),read(LIVE),read(HUMAN)
    frozen_items={r['segment_id']:r for r in frozen['items']};live_items={r['segment_id']:r for r in live['items']}
    frozen_rec={r['sample_id']:r for r in frozen['recordings']};live_rec={r['sample_id']:r for r in live['recordings']}
    if (len(frozen_items)!=730 or set(frozen_items)!=set(live_items) or any(frozen_items[k]['score']!=live_items[k]['score'] or frozen_items[k]['audit_rank']!=live_items[k]['audit_rank'] for k in frozen_items)
            or len(frozen_rec)!=82 or set(frozen_rec)!=set(live_rec) or any(frozen_rec[k]['score']!=live_rec[k]['score'] for k in frozen_rec)):
        raise ValueError('frozen score cohort changed')
    if not human['audio_dataset_signature']==frozen['metadata']['audio_dataset_signature']==live['metadata']['audio_dataset_signature']:
        raise ValueError('human snapshot audio/window signature mismatch')
    answers=human['answers'];ids=sorted(answers)
    if len(ids)!=60 or set(ids)!={r['segment_id'] for r in frozen['items'] if r['audit_rank'] is not None}:
        raise ValueError('authoritative human snapshot is not the frozen 60-case sample')
    if len({frozen_items[k]['sample_id'] for k in ids})!=60:raise ValueError('audit sample is not one segment per recording')
    if any(answers[k].get(field) not in {'good','bad'} for k in ids for field in ('judgement','first_judgement')):
        raise ValueError('human snapshot contains a nonbinary or missing judgement; no rows silently removed')
    if any(answers[k].get('first_label_blind') is not True or answers[k].get('latest_label_blind') is not True for k in ids):
        raise ValueError('blind-label provenance differs from the supplied audit snapshot')
    x=np.array([frozen_items[k]['score'] for k in ids],float)
    latest=np.array([answers[k]['judgement']=='good' for k in ids],int);first=np.array([answers[k]['first_judgement']=='good' for k in ids],int)
    if not np.isfinite(x).all() or any(answers[k]['machine_score_at_review']!=frozen_items[k]['score'] for k in ids):
        raise ValueError('human machine-score snapshot differs from frozen continuous score')
    latest_report=binary_metrics(x,latest);latest_report['bootstrap95']=bootstrap_intervals(x,latest,kind='binary')
    first_report=binary_metrics(x,first);first_report['bootstrap95']=bootstrap_intervals(x,first,kind='binary')
    recording_ids=sorted(frozen_rec);recording_scores=np.array([frozen_rec[sid]['score'] for sid in recording_ids],float)
    teacher,teacher_audit=load_teacher_scores(MANIFEST,recording_ids)
    crosschecks=[]
    for path in CROSSCHECKS:
        other,audit=load_teacher_scores(path,recording_ids,require_eligible=False)
        mismatch=int(np.sum(other!=teacher))
        crosschecks.append({'source':str(path.relative_to(ROOT)),'aligned_recordings':audit['aligned_recordings'],'score_mismatches':mismatch})
        if mismatch:raise ValueError('teacher score sources disagree')
    teacher_report=continuous_correlations(recording_scores,teacher)
    teacher_report.update({'primary_measure':'spearman','secondary_measure':'pearson','analysis_unit':'recording'})
    teacher_report['bootstrap95']=bootstrap_intervals(recording_scores,teacher,kind='continuous')
    ridge,ridge_values=ridge_oof_report(RIDGE,recording_ids,teacher)
    analysis_path=LP.OUT/'analysis.json'
    if ridge['available'] and analysis_path.exists():
        old_report=next(r for r in read(analysis_path)['models'] if r['model']=='Ridge_P5')
        ridge['previous_reported_repeatwise_spearman']=old_report['oof_spearman']
        ridge['previous_report_repeatwise_difference']=abs(ridge['repeatwise_spearman']['mean']-old_report['oof_spearman'])
    private_human=[{'segment_id':k,'sample_id':frozen_items[k]['sample_id'],'score':float(x[i]),'machine_binary50':'good' if x[i]>=50 else 'bad',
                    'human_latest':answers[k]['judgement'],'human_first':answers[k]['first_judgement'],'audio_source':answers[k]['audio_source'],
                    'first_label_blind':answers[k]['first_label_blind'],'latest_label_blind':answers[k]['latest_label_blind']} for i,k in enumerate(ids)]
    private_teacher=[{'sample_id':sid,'new_recording_reference_index':float(recording_scores[i]),'teacher_total_score':float(teacher[i]),
                      'coverage':frozen_rec[sid]['coverage'],'duration_s':frozen_rec[sid]['duration_s'],
                      'ridge_p5_mean_oof_prediction':float(ridge_values[i]) if ridge_values is not None else None} for i,sid in enumerate(recording_ids)]
    public={'created_utc':datetime.now(timezone.utc).isoformat(),'threshold_rule':{'good':'score >= 50','bad':'score < 50','threshold':50,'uncertain_class_used':False,
             'rule_origin':'binary reporting rule; frozen continuous scores and human labels unchanged'},
            'human_audit60':{'latest':latest_report,'first':first_report,'changed_judgements':int(np.sum(first!=latest)),
                            'sample_scope':'conditional on the deliberately selected original three-machine-state sample, one segment per recording; not all-730 population performance',
                            'original_sampling_state_counts':dict(Counter(frozen_items[k]['state'] for k in ids)),
                            'listening_source_counts':dict(Counter(answers[k]['audio_source'] for k in ids)),
                            'first_and_latest_blind_true':60},
            'new_recording_index_vs_teacher':teacher_report,'ridge_p5_saved_oof_vs_teacher':ridge,
            'teacher_source_audit':{'primary_source':str(MANIFEST.relative_to(ROOT)),**teacher_audit,'crosschecks':crosschecks},
            'score_integrity':{'frozen_vs_live_scores_identical':True,'frozen_vs_live_audit_ranks_identical':True,'no_models_trained':True,'human_sources_modified':False,
                               'teacher_grain':'82 distinct recordings, no duplication to 730 segments','recording_aggregation':'usable segment scores weighted by actual segment duration'},
            'source_sha256':hashes,'script_sha256':sha(__file__),
            'interpretation_notes':[
                'All human binary classification metrics use all 60 judged cases at the requested 50 threshold, not the previous 40-case clear-decision subset.',
                'Human Spearman and point-biserial Pearson correlate the continuous index with good=1 and bad=0, not with a human 0-100 scale.',
                'All raw p values are two-sided exploratory descriptions without multiplicity adjustment; these data and feature ideas have been examined repeatedly.',
                'Bootstrap95 uses 10000 paired recording resamples with seed 20260930, holding model outputs fixed. The 60-case intervals describe this conditional sample; they do not recover population performance or model-training uncertainty.',
                'The new index was trained against old listener good/bad labels, not teacher total scores. Teacher-total correlation is an auxiliary association, not teacher-grade accuracy.',
                'The old Ridge model targeted teacher totals with five features; the new index targets listener labels with thirteen features and different acoustic processing. Comparing correlations does not establish an improvement in the same prediction task.',
                'Most human reviews used denoised audio (59/60); model acoustics used original audio.',
                'Distinct recording IDs do not establish distinct speakers. No student text, notes or individual scores are in this public artifact.'
            ]}
    if any(sha(path)!=hashes[str(path)] for path in paths):raise ValueError('an input changed during analysis')
    save(PRIVATE/'human_alignment.json',private_human);save(PRIVATE/'teacher_alignment.json',private_teacher)
    save(PRIVATE/'source_snapshot.json',{'source_sha256':hashes,'created_utc':public['created_utc']})
    save(PUBLIC/'results.json',public)
    print(json.dumps({'complete':True,'human_n':60,'binary_accuracy':latest_report['accuracy'],'binary_balanced_accuracy':latest_report['balanced_accuracy'],
                      'human_auroc':latest_report['auroc'],'human_spearman':latest_report['spearman']['coefficient'],
                      'teacher_n':82,'teacher_spearman':teacher_report['spearman']['coefficient'],'ridge_oof_available':ridge['available']},ensure_ascii=True))


if __name__=='__main__':main()
