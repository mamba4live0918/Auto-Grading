# 本地语音边界标注网页

双击 [start.cmd](start.cmd)，或在论文项目根目录运行：

```powershell
python apps/audio_review/server.py --open
```

网页只在 `127.0.0.1:8765` 提供服务。它按 D01_1 至 D04_2、A01_1 至 A08_2 的顺序显示现有 24 段盲化音频。先听完 8 段 D，写下并固定判断规则，才能进入 16 段 A。界面不读取或展示隐藏映射、自动边界、模型结果、教师总分。

播放音频后，在波形上拖动选择可闻语音范围，再点“标为语音”或“标为不确定”。可在下方表格调整时间、增加说明。无语音的片段需明确勾选确认框。每次修改会自动保存到 `output/human_review_annotations/review_state.json`；点击“完成本段并下一段”会将该片段记为已审核。

“导出 CSV 记录”会下载 ZIP，并在 `output/human_review_annotations/` 写出 `boundaries.csv`、`review_status.csv`、`rules.txt`。CSV 列与原空白模板一致。未完成片段在状态表中为 `reviewed=no`，其临时边界不会进入导出的边界表。

本工具只记录**可闻语音边界**。表达质量分数需要先确定独立的评分维度和行为锚点，不能把边界标注当成质量评分。只有一位标注者时，结果属于单人人工核查。
