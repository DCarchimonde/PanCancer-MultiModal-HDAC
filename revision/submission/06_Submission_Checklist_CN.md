# BMC Bioinformatics 大修上传清单

投稿编号：`8b1e184c-443d-4707-954c-70a65930e16f`  
返修截止：2026-07-27  
证据快照：`fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`

## 需要上传

1. `01_Revised_Manuscript.pdf`：clean revised manuscript。
2. `01A_Revised_Manuscript_Reviewer_Marked.pdf`：给审稿人查看的红色修订稿。若投稿系统有 “annotated/marked manuscript” 或 “for reviewers” 槽位，单独上传到该槽位；不要把它标成 Additional file。
3. `02_Response_to_Reviewers.pdf`：17 条意见逐点回复。
4. `03_Additional_file_1_Supplementary_Material.pdf`：24 个 Supplementary table 主题与 6 个不重复的补充诊断图组。
5. `04_Additional_file_2_Revision_Tables.xlsx`：58 sheets（README、Sheet Index、Data Dictionary 与 55 个数据 sheets）的机器可读结果。
6. `05_LaTeX_Source.zip`：clean/marked 主稿、Supplement、回复信的 TeX 源码及编译所需图表。

如果系统要求单独上传图文件，可从源码包的 `manuscript/figures/` 与 `supplement/figures/` 选择对应 vector PDF；不要上传旧原稿中的模拟生存、模拟分期、旧 docking 或 GDSC 图。

## 不要上传

- `07_INTERNAL_Audit_Assessment_CN_DO_NOT_UPLOAD.md`：仅供作者和导师内部核对。
- `08_COMPLETE_Revision_Submission_Package.zip`：本地留存/传输总包；投稿系统通常应逐项上传其中正式文件，而不是再上传总包。
- `FINAL_QA_REPORT.md`：内部质量检查记录，除非编辑明确索要。
- 任何上一轮名称相同但日期或 hash 不同的旧投稿包。
- 旧 `Supplementary_Material.pdf`、旧 GDSC 图、旧模拟 survival/stage/TMB 图。

## 上传前作者必须人工确认

- [ ] 所有作者姓名、顺序、单位和通讯作者信息正确。
- [ ] Funding grant numbers：`823B2095`、`SYWD2024255`、`SYW2025185` 均准确。
- [ ] Authors' contributions 与实际贡献一致，所有作者同意。
- [ ] Competing interests、ethics、consent 和 data availability 声明准确。
- [ ] AI-assisted language/formatting disclosure 符合作者与期刊政策。
- [ ] 正文标题保持原投稿题目：`Multi-Modal Molecular Representation Learning Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal`。
- [ ] 上传界面的 Figure/Table/Additional file 标签与文件名一致。
- [ ] 回复信上传后抽查 17 个 Comment 标题是否全部可见。
- [ ] CPU 型号与 RAM 保持 `not recorded`，不要根据当前机器补猜。
- [ ] NSFC 编号 `823B2095` 已逐字符对照项目批件或导师确认；本清单不根据编号格式擅自修改。
- [ ] 若投稿系统没有 reviewer-marked manuscript 专用槽位，先按 decision letter/系统说明确认，不把红色稿误传为 Supplement。

## 关键科学口径

- dual-stream 与 fingerprint MLP 总体相当，不主张多模态优势。
- 模型仅输入化学结构，不预测指定 cell/dose/time 响应。
- measured LINCS 是实测表达证据，不是 viability、efficacy 或独立外部平台验证。
- Mocetinostat 为 core，NCH-51 为 secondary，TC-H-106 为 exploratory。
- RG2833 为 prediction-only；Tianeptinaline/BG-1010 因身份冲突排除。
- reversal-associated genes 不是 direct targets。
- HDAC class enrichment 仅在 signed wTRS 下显著，因此是 metric-dependent。
- enrichment consensus 对每个 metric 独立计算：4 个 generalization regimes × 2 models × 3 seeds × 22 cancers，共 528 个等权 within-library rank fractions/compound（0 = strongest）；0.5%、1%、5%、10% 是本次修回固定阈值，并非前瞻性预注册。Candidate stability 使用相反方向的 strength percentile（100 = strongest）。
- `00_TCGA_Cohorts` 列出 22 个 TCGA/GDC projects；重复出现的 disease-matrix 和 observed-mask 校验值已明确标为 global hashes，并非逐癌种文件 hash。逐癌种 tumor/normal counts 未保留，因此明确留空，未猜测。
- `21_Condition_Manifest` 明确披露分析时的 83,490-row condition-level CSV 及其 SHA256 未进入冻结便携包；没有编造校验值。
- g:Profiler 的真实 Entrez intersection genes 与 GO evidence codes 已分列，5,037 个 term 的重建交集计数均通过审计。
- DepMap 表明 HDAC3 是主要遗传依赖背景，但不等于候选药效或选择性。
- docking 是 protocol/receptor sensitivity，不证明 binding。
- Entinostat GDSC `R=-0.052, P=0.859` 是 null result，不是支持性验证。

## 提交后建议保存

- 提交成功页面和系统生成 PDF 的截图。
- 所有已上传文件的 SHA256（见 `FILE_SHA256SUMS.txt`）。
- 编辑系统的自动回执邮件。
- 本次完整 ZIP 与 GitHub evidence commit。
