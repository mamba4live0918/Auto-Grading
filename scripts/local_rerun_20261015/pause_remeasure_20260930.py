"""Post-hoc pause/filler analysis; originals and student-level data stay private.

Measurement runs under .venv-asr; this statistics runner uses .venv311.
The analysis rules are frozen in docs/research/2026-09-30-pause-remeasure-plan.md.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import sklearn

import verbatim_main as VM

import localpaths as LP
ROOT = Path(__file__).resolve().parents[2]
PRIVATE_IN = ROOT / 'data/private/pause_remeasure_20260930'   # 9/30 local silence measurements (read only)
PRIVATE = LP.RUN / 'pause_remeasure'
PUBLIC = LP.PUBLIC / 'pause_remeasure'
PLAN = ROOT / 'docs/research/2026-09-30-pause-remeasure-plan.md'
OLD_ROWS = LP.OUT / 'disfl_clip_rows.json'
OLD_ARRAYS = LP.OUT / 'disfl_main_arrays.npz'
FEATURES = LP.RUN / 'features_private.csv'
N_PERM, SEED_AUC, SEED_CV = 999, 20260929, 20260928
TOL = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def ready(v):
        if isinstance(v,dict):
            return {k:ready(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):
            return [ready(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):
            return None
        return v
    temp.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def holm(pvals):
    return VM.holm(pvals)


def single_metric(values, bad, n_perm=N_PERM):
    v, y = np.asarray(values, float), np.asarray(bad, int)
    if len(v) != len(y) or not np.isfinite(v).all():
        raise ValueError('missing or invalid values in requested metric; no silent case deletion')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('both outcome classes are required')
    auc = float(roc_auc_score(y, v))
    result = {'n': len(y), 'n_bad': int(y.sum()), 'auroc_predicting_bad': auc,
              'median_good': float(np.median(v[y == 0])), 'median_bad': float(np.median(v[y == 1]))}
    if n_perm:
        rng = np.random.default_rng(SEED_AUC)
        null = [roc_auc_score(rng.permutation(y), v) for _ in range(n_perm)]
        result['p_one_sided'] = float((1 + np.sum(np.asarray(null) >= auc - TOL)) / (n_perm + 1))
    return result


def join_rows(old, measured, expected=82):
    if len(old) != expected or len(measured) != expected:
        raise ValueError('unexpected clip cohort size')
    by_clip = {r['clip']: r for r in measured}
    if len(by_clip) != expected or len({r['clip'] for r in old}) != expected:
        raise ValueError('duplicate clip key')
    if set(by_clip) != {r['clip'] for r in old}:
        raise ValueError('clip cohort changed')
    out = []
    for row in old:
        new = by_clip[row['clip']]
        if new['sid'] != row['sid']:
            raise ValueError('clip-to-sid mapping changed')
        out.append({**new, 'j': row['j'], 'p5': row['p5']})
    return out


def cv_auc(X, y, reps=10):
    X, y = np.asarray(X, float), np.asarray(y, int)
    if not np.isfinite(X).any(axis=0).all():
        raise ValueError('all-missing feature column')
    if np.isinf(X).any():
        raise ValueError('infinite model input')
    if np.isnan(X).any():
        for repeat in range(reps):
            split = StratifiedKFold(5, shuffle=True, random_state=SEED_CV+repeat)
            for fold, (tr, _) in enumerate(split.split(X, y)):
                missing = np.flatnonzero(~np.isfinite(X[tr]).any(axis=0))
                if len(missing):
                    raise ValueError(f'all-missing training fold {repeat}:{fold}, columns {missing.tolist()}')
    # Exact station-6 folds, imputation, scaling, solver and regularisation.
    with threadpool_limits(limits=1):
        return VM.cv_auc(X, y, reps=reps)


_WORK = None


def _worker_init(base, extra, y, observed_base, reps):
    global _WORK
    _WORK = (base, extra, y, observed_base, reps)
    threadpool_limits(limits=1)


def _perm_chunk(task):
    start, indices = task
    base, extra, y, abase, reps = _WORK
    result = [float((cv_auc(np.c_[base, extra[index]], y, reps) - abase).mean()) for index in indices]
    return start, result


def model_gain(base, extra, y, n_perm=N_PERM, reps=10, workers=3, progress=None):
    base, extra, y = np.asarray(base, float), np.asarray(extra, float), np.asarray(y, int)
    if extra.ndim == 1:
        extra = extra[:, None]
    if len(base) != len(y) or len(extra) != len(y) or not np.isfinite(extra).all():
        raise ValueError('invalid added features')
    abase, aplus = cv_auc(base, y, reps), cv_auc(np.c_[base, extra], y, reps)
    observed = float((aplus - abase).mean())
    null = np.empty(n_perm)
    rng = np.random.default_rng(SEED_CV)
    indices = np.asarray([rng.permutation(len(y)) for _ in range(n_perm)])
    tasks = [(i, indices[i:i+20]) for i in range(0, n_perm, 20)]
    if workers == 1:
        _worker_init(base, extra, y, abase, reps)
        for task in tasks:
            start, values = _perm_chunk(task)
            null[start:start+len(values)] = values
            if progress:
                progress(min(start+len(values), n_perm), n_perm)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(base, extra, y, abase, reps)) as pool:
            jobs = [pool.submit(_perm_chunk, task) for task in tasks]
            done = 0
            for job in as_completed(jobs):
                start, values = job.result()
                null[start:start+len(values)] = values
                done += len(values)
                if progress:
                    progress(done, n_perm)
    result = gain_result(abase,aplus,null,len(y))
    return result, {'base_cv': abase, 'added_cv': aplus, 'null': null}


def gain_result(abase,aplus,null,n):
    observed = float((aplus-abase).mean())
    result = {'n': n, 'auroc_base_mean': float(abase.mean()),
              'auroc_added_mean': float(aplus.mean()), 'delta_mean': observed,
              'delta_repeat_range': [float((aplus-abase).min()), float((aplus-abase).max())],
              'p_one_sided': float((1 + np.sum(null >= observed-TOL)) / (len(null)+1)),
              'n_permutations': len(null)}
    return result


def _progress(name):
    last = [0.0]
    def callback(done, total):
        now = time.monotonic()
        if now-last[0] >= 20 or done == total:
            print(json.dumps({'phase': name, 'permutations_completed': done, 'total': total}), flush=True)
            last[0] = now
    return callback


def condition_matrix(rows, key, whole=False):
    field = 'whole_conditions' if whole else 'conditions'
    metrics = {name: [r[field][key][name] for r in rows] for name in
               ('mid_rate', 'sentence_end_rate', 'clause_boundary_rate', 'filler_rate', 'long_rate')}
    return metrics


def coverage(rows, key, whole=False):
    field = 'whole_conditions' if whole else 'conditions'
    values = [r[field][key] for r in rows]
    n = sum(v['n_pause'] for v in values)
    n_strict = sum(v['strict']['n_unresolved'] for v in values)
    return {'n_clips': len(values), 'n_internal_silences': n,
            'compat_unresolved': sum(v['n_unresolved'] for v in values),
            'strict_unresolved': n_strict,
            'strict_classifiable_coverage': (n-n_strict)/n if n else None,
            'n_windows_with_strict_missing_rate': sum(v['strict']['mid_rate'] is None for v in values),
            'n_fillers': sum(v['n_filler'] for v in values),
            'n_windows_with_zero_silent_pause': sum(v['n_pause'] == 0 for v in values),
            'min_phonation_s': min(v['phonation_s'] for v in values)}


def summary_markdown(report):
    lines = ['# 新版停顿与 um/uh 补测（2026-09-30）', '',
             f"已完成新版自动测量与原人工评价的补充分析：82份录音、82个原听辨窗口；主要统计固定{report['cohort']['n_good_bad']}段好／不好（好{report['cohort']['n_good']}、不好{report['cohort']['n_bad']}）。",
             '', '这次使用原声250ms停顿与本次原声转写；句中／句末／分句位置按旧实验可比的标点规则计算，um/uh等填充词来自CrisperWhisper。已试听反馈不等同于全部候选的人工真值。',
             '', '## 同一30秒听辨窗口的单项比较', '',
             '| 指标 | AUROC（较高值挑出“不好”） | 置换p | Holm校正p | 好／不好中位数（每发声分钟） |',
             '|---|---:|---:|---:|---:|']
    names = {'mid_rate': '句中静默', 'sentence_end_rate': '句末静默',
             'clause_boundary_rate': '分句边界静默', 'filler_rate': 'um/uh等填充词'}
    for k, v in report['main_single_metrics'].items():
        lines.append(f"| {names[k]} | {v['auroc_predicting_bad']:.3f} | {v['p_one_sided']:.3f} | {v['p_holm']:.3f} | {v['median_good']:.2f}／{v['median_bad']:.2f} |")
    lines += ['', 'AUROC为0.5表示没有排序区分力；高于0.5不自动代表证据充分。4项单项比较统一作Holm校正。',
              '', '## 加入原P5模型后', '', '| 增量 | 原模型AUROC | 加入后AUROC | 增量 | Holm校正p |', '|---|---:|---:|---:|---:|']
    for k, title in [('silence', '加入长停顿＋句中停顿'), ('filled', '再加入填充词')]:
        v = report['model_gains'][k]
        lines.append(f"| {title} | {v['auroc_base_mean']:.3f} | {v['auroc_added_mean']:.3f} | {v['delta_mean']:+.3f} | {v['p_holm']:.3f} |")
    lines += ['', '沿用原P5、L2逻辑回归C=1、10次分层5折；填缺与标准化仅在训练折拟合。两个增量各999次联合新增列置换，统一作Holm校正。该置换会破坏新增指标与P5的相关性，属于预定的边际打乱对照，不解释成严格控制P5后的条件独立检验。',
              '', '## 旧转写、降噪和300ms敏感性对照', '', '| 测量条件 | 句中AUROC | 句末AUROC | 分句AUROC | 填充词AUROC |', '|---|---:|---:|---:|---:|']
    for key, values in report['sensitivity'].items():
        lines.append('| '+key+' | '+' | '.join(f"{values[k]['auroc_predicting_bad']:.3f}" for k in names)+' |')
    lines += ['', '所有条件使用同一批片段，不从其中挑最好结果；本表仅描述。边界停顿合并对照及旧原实验数值见results.json。',
              '', '## 整段录音补充（仅描述）', '', '| 指标 | AUROC |', '|---|---:|']
    for k in names:
        lines.append(f"| {names[k]} | {report['whole_descriptive']['single_metrics'][k]['auroc_predicting_bad']:.3f} |")
    whole = report['whole_descriptive']['models']
    lines += ['', f"整段P5／加静默指标／再加填充词的交叉验证AUROC：{whole['p5']:.3f}／{whole['plus_silence']:.3f}／{whole['plus_filled']:.3f}。标签仍来自原30秒试听，不能称作整段人工评价。",
              '', '## 测量覆盖与解释限制', '']
    cov = report['coverage']['250_original']
    pct = '无静默' if cov['strict_classifiable_coverage'] is None else f"{100*cov['strict_classifiable_coverage']:.1f}%"
    lines += [f"严格词时间对齐在全部82个L窗口的覆盖率为{pct}；{cov['n_windows_with_strict_missing_rate']}个窗口含未对齐静默。严格率保留缺失，不当成零，不进入本轮主模型。主结果采用可比旧规则，依然可能受标点、远邻词及词时间跨越影响。",
              '', '发声时间由原声Praat sounding区间直接测量，包括没有静默停顿的片段；没有猜分母。静默和填充词可能重叠，不相加成真实中断总数。',
              '', '本轮是在旧结果已知后、同一批数据上开展的测量补充。新停顿为本机Praat6.1.38，原P5来自10/15本机重新运行的Praat6.1.38；测量版本及输入哈希均保留，不能把新结果当作独立复现或仅由ASR造成的改善。',
              '', '页内未导出的新勾选没有进入人耳准确率计算。原50处旧位置标签的可用性、输入审计及独立重算另见verification.json；不以口头反馈宣称所有候选正确。',
              '', '## 复算文件', '', '- results.json：公共统计汇总。', '- configuration.json：计划、输入及代码哈希和软件版本。',
              '- verification.json：独立核验。', '- data/private/pause_remeasure_20260930/：逐片段特征、标签关联及置换数组，仅本地保留。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume-computed-permutations',action='store_true',
                        help='recover the just-computed NPZ after a final output write failure; validates inputs/CV before reuse')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    old_files = [OLD_ROWS, OLD_ARRAYS, FEATURES,
                 ROOT/'output/disfluency_20260928/结果摘要.md', ROOT/'output/verbatim_20260929/main.json']
    before = {str(p.relative_to(ROOT)): sha256(p) for p in old_files}
    measurement_path = PRIVATE_IN/'features.json'
    measurement = json.loads(measurement_path.read_text(encoding='utf-8'))
    old_rows = json.loads(OLD_ROWS.read_text(encoding='utf-8'))
    rows = join_rows(old_rows, measurement['rows'])
    gb = [r for r in rows if r['j'] in ('好', '不好')]
    if len(gb) != 65:
        raise ValueError('original 65 labelled cohort changed')
    y = np.array([r['j'] == '好' for r in gb], int)
    bad = 1-y
    X5 = np.array([r['p5'] for r in gb], float)
    with np.load(OLD_ARRAYS) as old:
        if not np.array_equal(old['y'], y) or not np.allclose(old['X5'], X5, equal_nan=True):
            raise ValueError('original label order/P5 arrays are not reproduced')
        old_a5 = old['a5'].copy()
        old_d3 = old['XD'][:, 1].copy()
    new_a5 = cv_auc(X5, y)
    if not np.allclose(old_a5, new_a5, atol=.005, rtol=0):
        raise ValueError('old P5 baseline not reproduced under same folds')
    key = '250_original'
    cols = condition_matrix(gb, key)
    main_columns = ('mid_rate', 'sentence_end_rate', 'clause_boundary_rate', 'filler_rate')
    report = {'analysis_type': 'post_hoc_measurement_sensitivity', 'cohort': {
        'n_recordings': 82, 'n_windows': 82, 'n_good_bad': 65, 'n_good': int(y.sum()), 'n_bad': int(bad.sum())},
        'main_single_metrics': {k: single_metric(cols[k], bad) for k in main_columns}}
    adjusted = holm({k:v['p_one_sided'] for k,v in report['main_single_metrics'].items()})
    for k,v in report['main_single_metrics'].items():
        v['p_holm'] = adjusted[k]
    report['old_station6_d3_descriptive'] = single_metric(old_d3, bad, n_perm=0)
    report['sensitivity'], report['coverage'] = {}, {}
    for condition in sorted(gb[0]['conditions']):
        values = condition_matrix(gb, condition)
        report['sensitivity'][condition] = {k: single_metric(values[k], bad, n_perm=0) for k in main_columns}
        boundary = np.asarray(values['sentence_end_rate'])+np.asarray(values['clause_boundary_rate'])
        report['sensitivity'][condition]['boundary_combined_rate'] = single_metric(boundary, bad, n_perm=0)
        report['coverage'][condition] = coverage(rows, condition)
    Xsilence = np.c_[cols['long_rate'], cols['mid_rate']]
    Xfill = np.asarray(cols['filler_rate'])[:, None]
    report['model_gains'] = {}
    print(json.dumps({'phase':'single_metrics_done','n':65}), flush=True)
    arrays = {'y': y, 'X5': X5, 'Xsilence': Xsilence, 'Xfill': Xfill}
    resumed = None
    if args.resume_computed_permutations:
        with np.load(PRIVATE/'analysis_arrays.npz') as saved:
            resumed = {k:saved[k] for k in saved.files}
        for k,value in arrays.items():
            if not np.array_equal(value,resumed[k],equal_nan=True):
                raise ValueError('computed permutation checkpoint has different inputs')
    for name, base, added in [('silence',X5,Xsilence), ('filled',np.c_[X5,Xsilence],Xfill)]:
        if resumed is None:
            result, values = model_gain(base, added, y, workers=args.workers, progress=_progress(name))
        else:
            abase,aplus = cv_auc(base,y),cv_auc(np.c_[base,added],y)
            stored_base = resumed['cv_base' if name=='silence' else 'cv_silence']
            stored_plus = resumed['cv_silence' if name=='silence' else 'cv_filled']
            null = resumed['null_'+name]
            if (len(null)!=N_PERM or not np.isfinite(null).all() or
                    not np.allclose(abase,stored_base,atol=1e-12,rtol=0) or
                    not np.allclose(aplus,stored_plus,atol=1e-12,rtol=0)):
                raise ValueError('completed checkpoint does not reproduce current CV results')
            result = gain_result(abase,aplus,null,len(y))
            values = {'base_cv':abase,'added_cv':aplus,'null':null}
        report['model_gains'][name] = result
        arrays['null_'+name] = values['null']
        if name == 'silence':
            arrays['cv_base'], arrays['cv_silence'] = values['base_cv'], values['added_cv']
        else:
            arrays['cv_filled'] = values['added_cv']
    adjusted = holm({k:v['p_one_sided'] for k,v in report['model_gains'].items()})
    for k,v in report['model_gains'].items():
        v['p_holm'] = adjusted[k]
    p5_names = ['articulation_rate','pause_rate_per_min','mean_pause_s','f0_sd_st','intensity_sd_db']
    with FEATURES.open(encoding='utf-8-sig') as f:
        whole_p5 = {r['sample_id']:[float(r['whole__'+k]) if r['whole__'+k] else np.nan for k in p5_names]
                    for r in csv.DictReader(f)}
    X5whole = np.array([whole_p5[r['sid']] for r in gb], float)
    wcols = condition_matrix(gb, key, whole=True)
    Xsilencewhole = np.c_[wcols['long_rate'],wcols['mid_rate']]
    Xfillwhole = np.array(wcols['filler_rate'])[:, None]
    report['whole_descriptive'] = {'single_metrics': {k: single_metric(wcols[k],bad,n_perm=0) for k in main_columns},
        'models': {'p5':float(cv_auc(X5whole,y).mean()),
                   'plus_silence':float(cv_auc(np.c_[X5whole,Xsilencewhole],y).mean()),
                   'plus_filled':float(cv_auc(np.c_[X5whole,Xsilencewhole,Xfillwhole],y).mean())},
        'coverage':coverage(gb,key,whole=True)}
    arrays.update(X5whole=X5whole,Xsilencewhole=Xsilencewhole,Xfillwhole=Xfillwhole)
    report['old_files_unchanged'] = all(sha256(ROOT/p) == h for p,h in before.items())
    if not report['old_files_unchanged']:
        raise ValueError('old source/result files changed during analysis')
    config = {'plan_sha256':sha256(PLAN), 'measurements_sha256':sha256(measurement_path),
              'input_sha256':before, 'code_sha256': {p.name:sha256(p) for p in [Path(__file__),Path(VM.__file__)]},
              'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
              'n_permutations':N_PERM,'cv_seed':SEED_CV,'single_metric_seed':SEED_AUC,
              'completed_permutations_recovered_after_json_write_fix':args.resume_computed_permutations,
              'measurement_versions':measurement.get('metadata',{}).get('versions',{})}
    np.savez_compressed(PRIVATE/'analysis_arrays.npz', **arrays)
    write_json(PRIVATE/'assembled_rows.json', rows)
    write_json(PUBLIC/'configuration.json',config)
    write_json(PUBLIC/'results.json',report)
    (PUBLIC/'结果摘要.md').write_text(summary_markdown(report),encoding='utf-8')
    print(json.dumps({'phase':'complete','n_good_bad':65,'old_files_unchanged':True}),flush=True)


if __name__ == '__main__':
    main()
