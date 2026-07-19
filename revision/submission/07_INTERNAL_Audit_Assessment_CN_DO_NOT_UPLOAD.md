# 投稿前审计、断线恢复与最终处理（内部文件，请勿上传期刊）

## 最终判断

第二对话框的审计大方向正确，尤其准确发现了 S23 对审稿意见的错误复述、g:Profiler `intersection_genes` 字段错误、HDAC 富集聚合描述缺失和 Figure 3 图注不符。上述实质问题均已修复并重新生成正文、Supplement、工作簿和回复信。

两项按作者要求保持不变：

- 标题保持原投稿题目：**Multi-Modal Molecular Representation Learning Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal**。
- AI-assisted technologies 声明保持上一版原文，未扩大或缩小。

标题可以保留，因为 `prioritizes` 描述的是计算排序，而不是宣称 dual-stream 优于 fingerprint MLP；signed wTRS 下的正式 class I HDAC 富集为标题提供统计依据。摘要、结果和讨论同时明确：该富集具有 metric dependence，双流模型没有稳定优势，也不推断疗效。

## GitHub 断线一致性核查

断线时 `major-revision-2026` 分支的 `revision/submission/` 只有 01--06 和内部 manifest；并没有实际提交 07 内部审计文件与 08 总包。因此它当时不等同于声称的“八件套”。本轮没有沿用旧 ZIP，而是由修正后的源码重新编译并生成 01--08，再覆盖同步至 GitHub。

## 第二对话框审计逐项结论

| 审计点 | 判断 | 最终处理 |
|---|---|---|
| S23 称湿实验“未被要求” | 正确发现事实错误 | 改为：未开展新湿实验；审稿人同时给出的 public transcriptomic validation 选项已用官方条件 measured LINCS 回应 |
| `intersection_genes` 实为 GO evidence codes | 正确，且会影响 redundancy filtering | 按冻结 API response 的 mapped-query 顺序重建真实 Entrez intersections；evidence codes 单列；5,037/5,037 term 的重建计数与服务返回 `intersection_size` 一致；代表通路和热图重算 |
| HDAC enrichment 聚合不清楚 | 正确 | 对每个 metric，在 4 splits × 2 models × 3 seeds = 24 runs 及 22 cancers 内分别排序并转 library percentile；每个化合物等权平均 528 个 run--cancer percentiles，升序定义 consensus rank |
| Figure 3 声称有 error bars | 正确 | 改为 OLS 黑线仅用于视觉概括；点级误差条未显示；相关置信区间来自 10,000 次 crossed candidate--cancer bootstrap |
| `hard-coded` 措辞风险 | 正确 | 改为“unsupported legacy value not linked to a reproducible run artifact”，旧 0.2841 不进入修订证据 |
| HDAC holdout 需注明 matched LINCS profiles | 正确 | 写明 39 个 annotation-defined library structures 中 30 个有 matched modeling profiles，共 1,856 signatures |
| OOD 用词过强 | 正确 | 改为 generalization/distribution-shift regimes，并明确不是每个候选在每个 regime 都未见过 |
| class cutoffs 称 `prespecified` | 有风险 | 改为 fixed revision cutoffs，并明确 not prospectively preregistered |
| 11,144 与 11,154 需映射审计 | 正确 | 前者是 submitted unique measured Entrez IDs，后者是 service effective domain；新增 `41A_gProfiler_Audit` sheet 与 JSON/CSV 审计 |
| 工作簿缺 21 号 sheet | 正确 | 新增 `21_Condition_Manifest`，记录 83,490-row condition table 的路径、schema、hash 与不重复嵌入的原因 |
| Sheet Index 的 Columns 为小数 | 正确 | 改为整数格式 |
| 网络仍可再放大 | 可选，不是阻断项 | 主文只显示 weighted-degree 最高的可读子集；Supplement 分两页扩展；完整 nodes/edges 在工作簿 |
| AI 声明建议扩大 | 作者明确不同意修改 | 保持上一版声明原文；作者仍须确认符合期刊政策并对声明负责 |
| Funding 编号格式异常 | 只能人工确认 | 不擅自修改；上传前逐字符核对 `823B2095`、`SYWD2024255`、`SYW2025185` |

## 完整任务线状态

- E1 真实 measured LINCS：完成。
- E2 冻结候选后的网络、通路、DepMap：完成。
- E3 controlled docking：完成。
- C4 直接外部药敏：候选匹配公开数据条件不满足；以 unavailable/null result 关闭，不伪装为验证。
- A 删除、降级和纠正旧分析：投稿材料、README、tracker 与旧入口已同步清理。
- B 主稿、图、Supplement、工作簿、复现信息与 17 条回复：完成并交叉检查。

## 原投稿图片的去向

原 PDF 有六个编号图组、七个嵌入栅格对象，但附件没有提供独立图源。它们没有被无理由丢弃：可保留的视觉问题已用修复后的冻结数据重画；依赖模拟或错误证据的旧像素图不能继续使用。

| 原图 | 最终处理 |
|---|---|
| Figure 1 pan-cancer heatmap | 保留视觉问题，以修复后 screening 数据重画为新 Figure 1A |
| Figure 2 top-10 wTRS distribution | 保留跨癌种分布叙事，按冻结候选、signed wTRS 与正式富集重画为新 Figure 1B--C |
| Figures 3--4 network/enrichment/survival | 删除模拟 survival 与 treated/untreated 含义；网络和通路以新 Figures 6--7 与 Supplementary Figure S4 重建 |
| Figure 5 TC-H-106 单一 docking | 由含 cocrystal redocking、positive control、decoy、三 seeds 与 4BKX sensitivity 的新 Figure 9/S6 替换 |
| Figure 6 HDAC1 DepMap + GDSC | 用 DepMap 26Q1 全 class I HDAC 结果替换；不显著 GDSC 仅保留为 null availability record |

主文 13 个图资产组成九个 figure groups；Supplement 六个图资产组成六个互补诊断 groups。两组文件的 SHA-256 交集为零。Supplement 展开的是 seed、condition、legacy sensitivity、expanded network、lineage 和 pose-level 诊断，不是重复粘贴主图。

## 最终文件核查摘要

- 主稿：36 页，九个主图组，36 条参考文献；原上传 TeX 同样是 36 条参考文献，并非由 38 减至 16。
- 回复信：10 页，17/17 comments，每条含 Response、Action/result、Revised text 与最终页码/行号位置。
- Additional file 1：32 页，S1--S24 表主题与 S1--S6 互补图组。
- Additional file 2：57 sheets，即 README、Sheet Index、Data Dictionary 与 54 个数据 sheets；公式错误扫描和绝对路径扫描均为零。
- Source ZIP：从独立目录成功编译主稿、Supplement 与回复信；不含 `.aux/.log/.fls/.fdb_latexmk` 临时文件。

## 仍需作者人工确认

以下内容不能由数据审计代替作者确认：作者姓名/顺序/单位/通讯邮箱、Funding 编号、作者贡献、利益冲突、伦理与 AI 声明是否符合真实情况和期刊政策。

任何人都不能保证编辑一定 accept。最终包能够保证的是：已知的数据、统计、术语、图表、编号、来源和跨文件一致性问题已被系统检查；不支持的结论已删除或降级；无法从冻结数据恢复的内容没有被编造。
