# Reproducibility and artifact provenance

This file defines the reproducibility boundary for the manuscript **A Leakage-Aware and Auditable Framework Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal**. It distinguishes preserved evidence from executable code and avoids implying that every stage can be reproduced bit for bit from GitHub alone.

## Direct answer

The complete revised workflow is **not a self-contained, bitwise-reproducible workflow from the archived GitHub code and currently downloadable public inputs alone**. The analysis code is archived, but the portable repository excludes large public source files, expression caches, GPU checkpoints, screening prediction arrays, and several analysis-time intermediates. Some current public holdings can also differ from the frozen versions used in the analysis.

The final submission is nevertheless auditable at the reported-result level:

- the clean and reviewer-marked manuscripts, Supplement, response, and machine-readable workbook can be rebuilt or inspected from the frozen submission source;
- final numerical summaries used in the manuscript are preserved in machine-readable tables;
- final figure assets are preserved as vector PDFs;
- run settings, seeds, reported runtimes, selected checkpoint hashes, software versions, and known omissions are recorded;
- the first-round submission snapshot is preserved at commit `0eb798f` and tag `round1-submitted-2026-07-25`;
- the corrected analysis evidence snapshot is commit `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`.

## Artifact classes

### 1. Original or legacy outputs retained for provenance

Legacy files such as `results/revision/Top5_Drugs_All_Cancers_Analysis.csv` and `results/revision/all_drug_names_check.csv` are retained to document the history of the project. They are not used to support the revised inferential claims. Analyses explicitly retired from inference are listed in `revision/analysis_scope_and_exclusions.md`.

### 2. Analyses regenerated during the first revision

The first revision regenerated or newly assembled the leakage-aware model summaries, corrected annotation-defined HDAC holdout, formal enrichment, candidate stability, predicted--measured comparison, structural-neighbor audit, frozen candidate evidence tiers, network/pathway analysis, DepMap context, and docking controls. Final portable outputs are represented in:

- `results/revision/final_statistics/`;
- `results/revision/reversal_gene_networks_pathways/`;
- `revision/rework_20260720/supplement/generated/`;
- `revision/submission/04_Additional_file_2_Revision_Tables.xlsx`;
- the first-round manuscript and Supplement figure assets.

The second revision changes framing, terminology, citations, metric qualification, and repository documentation. It does not retrain a model, rescreen the library, or add a post hoc magnitude-adjusted endpoint.

### 3. Preserved submission evidence

The following are preserved and can be checked without rerunning the neural models:

- all 34 reported run-level benchmark rows and their aggregate/paired summaries;
- corrected-HDAC composition and pre/post-correction audit tables;
- formal class-enrichment results for each metric and cutoff;
- candidate-stability, predicted--measured bootstrap, leave-one-out, identity, structural-neighbor, pathway, network, DepMap, docking, and null/sensitivity tables in Additional file 2;
- final main and Supplementary figure PDFs;
- frozen STRING and g:Profiler response caches used for the revised network/pathway audit;
- the PDB source structures used for docking controls;
- LaTeX source and generated table fragments needed to compile the submission documents.

These artifacts support verification of the reported values and recompilation of the documents. Their presence is not equivalent to an executable archive of every upstream computation.

### 4. Large or locally held artifacts not redistributed in Git

The `.gitignore` intentionally excludes:

- LINCS/GEO GCTX matrices and large official metadata files;
- TCGA expression downloads and locally prepared disease matrices;
- expression caches (`.npy`, `.npz`, and memory-mapped arrays);
- model checkpoints (`.pt`, `.pth`, and `.ckpt`);
- screening prediction and score arrays;
- large raw or external datasets and runtime logs.

The expected large-input layout is summarized in `data/README.md`. Users must obtain the public source data under the terms of GDC/TCGA, GEO/LINCS, DepMap, RCSB PDB, STRING, and g:Profiler. Re-downloading a public resource does not by itself guarantee byte-identical reconstruction of the frozen analysis input.

### 5. Intermediates no longer available

The following omissions are explicitly known:

- the analysis-time 83,490-row condition-level source table and its SHA-256 hash were not retained in the portable archive;
- final per-cohort TCGA tumor/normal counts were not preserved and were not reconstructed from current GDC holdings;
- the CPU model and RAM used for the frozen AutoDL analyses were not captured;
- large GPU checkpoints, expression caches, screening arrays, and multiple upstream result directories required by some scripts are absent from Git.

No replacement values or retrospective checksums are fabricated for these items.

## Stage-by-stage status

| Stage | Code archived | Principal public input | Frozen result evidence preserved | Re-executable from Git + current public inputs alone | Boundary |
|---|---:|---|---:|---:|---|
| TCGA disease-signature preparation | Yes | GDC/TCGA | Global matrix and observation-mask hashes; final disease-dependent summaries | No | Download snapshot and final per-cohort counts are not fully archived. |
| LINCS quality control and expression cache | Yes | GEO GSE92742 and official LINCS metadata | Dataset-composition and condition-coverage tables | No | GCTX, official large files, local master table, and expression cache are excluded. |
| Entity-aware split generation | Yes | Curated LINCS master table | Pair split files and full split inventories in Additional file 2 | Not exactly | The curated master table and complete generalization split files are not all versioned. |
| Neural training and repeated evaluation | Yes | Prepared cache and split files | All 34 reported run summaries, paired summaries, settings, runtimes, and selected checkpoint hashes | No | Checkpoints, cache arrays, and several benchmark directories are absent. |
| 28,477-compound screening | Yes | Screening library, disease matrices, checkpoints | Final enrichment and candidate-stability tables and figures | No | Checkpoints and run-level score arrays are absent. |
| Official-condition measured LINCS comparison | Yes | Official LINCS metadata and measured expression cache | Candidate condition coverage, aggregates, bootstrap and leave-one-out tables | No | Expression cache and the 83,490-row analysis-time condition table are absent. |
| Structural-neighbor and applicability audit | Yes | Curated library and split inventories | Final candidate and split audit tables | Partially | Final evidence is preserved, but not every upstream split/library artifact is versioned. |
| Network and pathway annotation | Yes | STRING/g:Profiler | Frozen API responses, corrected enrichment tables, and final figures | Partially | Frozen responses are preserved; service-side reruns may change over time. |
| DepMap context | Yes | DepMap Public 26Q1 | Final pan-cancer, lineage, heterogeneity, and correlation tables in the submission package | No | The source subset and analysis directory are not in the portable Git tree. |
| Docking controls | Yes | RCSB PDB 4LXZ and 4BKX | Source structures, final tables, and figure assets | Not bitwise | Docking binaries/potentials and stochastic outputs require environment reconstruction; run directories are not fully archived. |
| Statistical/document assembly | Yes | Frozen derived tables and figure assets | Clean/marked manuscript, Supplement, response, workbook, and LaTeX source | Yes for document compilation and result inspection | This recompiles preserved evidence; it does not recompute the upstream neural workflow. |

## Practical use

For review or audit, use the frozen submission files and Additional file 2 as the primary machine-readable result record. To rebuild the documents, use the LaTeX source bundle under `revision/second_revision_20260824/` and its bundled figure/table assets. To rerun an upstream analysis stage, first supply every input listed by that script and compare file versions and hashes with the retained manifests; do not assume that missing artifacts can be reconstructed exactly from current public downloads.

## Reporting rule

Use the following wording when describing this repository:

> The repository archives the analysis code, retained derived evidence, and complete document source. Reported values can be audited against machine-readable tables and the submission documents can be recompiled. A self-contained bitwise rerun of every training and screening stage from GitHub and current public inputs alone is not guaranteed because large run artifacts and specified analysis-time intermediates were not retained.
