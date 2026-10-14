# 论文分析脚本（本机运行，2026-10-15）

论文报告的数字由这个目录里的脚本在本机算出。学生数据不在本仓库中；脚本读取的私有输入在下面列出。

## 环境与输入

- Python 3.11.9（`.venv311`）：numpy 2.4.6、scipy 1.17.1、scikit-learn 1.9.0、pandas 2.3.3、webrtcvad-wheels 2.0.14、opensmile 2.6.0（openSMILE 3.0.1）。
- Praat 6.1.38：parselmouth 0.4.7（装在 `.venv-asr`，`localpaths.py` 会把它加进搜索路径）。
- 学生音频：`artifacts/research/staging/notebook-01-local-v1/work/<编号>/audio/analysis.wav`（16 kHz 单声道 PCM16，用 ffmpeg 8.1.1 转换）。
- 专业参照：`D:\ThesisReferenceData\europarl_asr_v1.0\selected_audio.v2\`（500 段发言和选取清单）。
- 转写 A：`artifacts/research/a4a7e64…/transcripts.jsonl`；E1 的音节核对用同一配置的另一次运行 `.worktrees/europarl-500/artifacts/research/444eb952…/transcripts.jsonl`。
- 老师总分：冻结清单 `.worktrees/europarl-500/artifacts/research/444eb952…/manifest.csv`。
- 听辨、标注、耳朵核对的答案和片段时间：`data/private/listening_20260926/`、`pause_check_20260928/`、`literature_cues_20260929/`；E16 的 60 段新听评答案在本机 `Downloads` 文件夹。
- 输出：逐份的私有结果在 `data/private/local_rerun_20261015/`，汇总在 `output/local_rerun_20261015/`；专业参照的 180 秒音频和 Praat 中间文件在 `D:\ThesisLocalRerun_20261015\`。

## 说明

1. `features.py`：用 parselmouth 里的 Praat 6.1.38 运行 `extract.praat`；WebRTC VAD 用 webrtcvad-wheels 2.0.14 里的 `_webrtcvad`。
2. eGeMAPSv02 用 openSMILE 的 Python 包计算（`run_extract_local.py`）。
3. `analysis.py` 不含论文里没有报告的一个对照模型。
4. `permutation.py` 用 9 个进程；种子按块固定，进程数不影响结果。

`verbatim.py`、`verbatim_check.py` 只作为模块被 `verbatim_main.py` 等引用，**不要单独运行 `verbatim_check.py`**。E1 的开头 48 秒、E4 的描述和 E13 的“只换转写”诊断是按保存的数字重建的脚本，运行前都核对过能复现保存的数字；E4 理由统计的关键词按原来的计数重建，所以理由表的计数只是同一统计的重现，不是独立的核对。

## 运行顺序（全部用 `.venv311/Scripts/python.exe`，在仓库根目录运行；脚本都在 `scripts/local_rerun_20261015/`）

| 步骤 | 命令 | 对应分析 |
|---|---|---|
| 1 | `test_features.py`、`synthetic_detectors.py` | 合成信号测试；Praat 与 WebRTC 在合成停顿上的对照 |
| 2 | `run_extract_local.py` | 82 份学生录音的特征和 eGeMAPS |
| 3 | `run_ref_local.py main`，再 `run_ref_local.py full` | 500 位专业发言人（开头 180 秒、全长） |
| 4 | `analysis.py`、`permutation.py`、`verify.py`、`posthoc_first48_local.py` | E1 |
| 5 | `make_tables_local.py` | 生成后面几步读的两张特征表（并核对音频哈希） |
| 6 | `ref_analysis.py`、`ref_analysis_full.py` | E2、E3、E10（全长参照） |
| 7 | `label_analysis.py data/private/listening_20260926/听辨结果_2026-09-28.csv data/private/listening_20260926/listening_key_DO_NOT_OPEN.csv data/private/local_rerun_20261015/students_ref.npz > data/private/local_rerun_20261015/out/label_analysis.json`；全长版把 `students_ref.npz` 换成 `out_students_ref_full.npz`，输出 `label_analysis_full.json` | E4 |
| 8 | `pause_check_local.py` | E5 |
| 9 | `disfl_dev.py`、`disfl_main.py`、`disfl_verify.py`、`disfl_whole.py` | E6 |
| 10 | `listening_descriptive_local.py`（读第 9 步的 `disfl_clip_rows.json`） | E4 的描述 |
| 11 | `litcues_main.py` | E7 |
| 12 | `teacher_score_check.py` | E8 |
| 13 | `closeness_check.py` | E9 |
| 14 | `whole_rerun.py`；`whole_verify.py > data/private/local_rerun_20261015/out/whole_verify.json` | E10 |
| 15 | `verbatim_main.py`、`verbatim_verify.py` | E12 |
| 16 | `pause_remeasure_20260930.py --workers 8`、`transcript_model_diagnostic.py` | E13 的模型部分 |
| 17 | `report_correlations_20260930.py` | E16 |
| 18 | `findings_bounds_20261012.py` | 事后补算的区间 |

运行时设 `PYTHONIOENCODING=utf-8`，否则重定向到文件的中文会乱码。E11、E13 的转写和停顿重测、E14、E15 的脚本不在这个目录里。
