# Multi-Modal Molecular Representation Learning for Pan-Cancer Transcriptomic Reversal

This repository contains the corrected major-revision analysis for:

> **Multi-Modal Molecular Representation Learning Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal**

The revision is a hypothesis-generating computational study. It separates model generalization, experimentally measured transcriptomic reversal, candidate identity, biological context, and docking sensitivity. It does **not** claim drug efficacy, clinical benefit, direct target discovery, biochemical target engagement, or superiority of the dual-stream model.

## Frozen evidence snapshot

- Revision branch: `major-revision-2026`
- Corrected analysis snapshot: `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`
- Perturbation profiles: 55,695 LINCS L1000 Level 5 profiles
- Output genes: 12,328
- Screened canonical compounds: 28,477
- Disease signatures: 22 TCGA cancer types
- Neural hardware recorded by run manifests: NVIDIA GeForce RTX 4090
- CPU model and RAM: not captured in the frozen AutoDL manifests and therefore not inferred

Large public source datasets are not redistributed because of file size and source-specific access terms. The repository contains scripts, checksums, manifests, plotting data, and redistributable processed outputs.

## What the models estimate

Both predictors receive chemical structure only. They do not receive cell-line identity, dose, exposure time, or other experimental-condition covariates. Their predictions are therefore structure-conditioned central tendencies across training conditions, not condition-specific responses.

The evaluated models are:

- `dual_stream`: atom-token Transformer stream fused with a Morgan-fingerprint stream;
- `fingerprint_mlp`: an internal Morgan-fingerprint multilayer perceptron baseline.

Across pair, leave-drug, leave-cell-line, scaffold, and corrected annotation-defined HDAC holdout evaluations, the models were broadly comparable. Paired confidence intervals did not establish a stable dual-stream advantage. The old claims of direct comparison with DeepCE or ChemCPA were invalid because the legacy scripts contained non-equivalent proxy implementations; those entry points now remain only as deprecation notices.

## Leakage-aware evaluation

The revision reports five complementary settings:

1. drug--cell pair split;
2. leave-drug-out;
3. leave-cell-line-out;
4. Murcko-scaffold split;
5. corrected annotation-defined HDAC holdout.

The corrected HDAC holdout contains 53,839 training profiles and 1,856 test profiles from 30 held-out canonical structures and 55 test cell lines. Exact test structure, drug identifier, signature, and drug--cell pair overlap are zero. Eleven test scaffolds occur in training and are disclosed; this is a class holdout, not a claim of universal scaffold independence. The discarded pre-correction summary contained 1,565 profiles from 21 structures and is retained only as provenance.

## Main findings and evidence boundaries

### Screening and class enrichment

Class I HDAC compounds are formally enriched at prespecified top 0.5%, 1%, 5%, and 10% cutoffs under signed wTRS after multiplicity correction. The same class enrichment is not significant under Spearman reversal. The result is therefore **metric-dependent**, not a universal HDAC-class effect.

Primary candidate stability uses 48 configurations:

`4 splits × 2 models × 3 seeds × 2 primary metrics`.

The 72-configuration result that additionally includes legacy wTRS is retained as a sensitivity analysis only.

### Candidate hierarchy

| Candidate | Revised role | Interpretation |
|---|---|---|
| Mocetinostat | Core computational candidate | Best combined revised evidence tier; hypothesis for follow-up |
| NCH-51 | Secondary candidate | Stable transcriptomic signal with residual scaffold exposure |
| TC-H-106 | Exploratory/original-method-sensitive | Downgraded because of close training-neighbor exposure |
| RG2833 | Prediction-only | No measured LINCS signature in the frozen cache |
| Tianeptinaline/BG-1010 | Identity-conflict excluded | Exact training exposure and unresolved name--structure conflict |

Belinostat, PCI-24781, and Panobinostat provide class-support context; Entinostat and Vorinostat are reference controls. These roles are evidence tiers, not claims of therapeutic efficacy.

### Measured LINCS expression

Measured reversal uses official LINCS condition metadata and distinguishes all-profile, QC-pass, and `is_hiq` strata. Profiles are aggregated within candidate/dose/time/official-cell conditions before candidate--cancer summaries. This is experimentally measured expression evidence, but it is not viability, efficacy, binding, or validation on an independent experimental platform.

The prediction--measurement analysis uses a two-way candidate-by-cancer bootstrap (10,000 iterations; seed 20260719) plus leave-one-candidate/cancer sensitivity. A fully condition-matched non-HDAC null could not be recovered from the archived official-condition outputs. An unmatched 11,445-compound all-profile rank analysis is provided only as sensitivity evidence.

### Biological context

- Network and enrichment nodes are **reversal-associated genes**, not direct drug targets.
- The custom g:Profiler request submitted 11,144 measured Entrez identifiers; the service reported an effective mapped domain of 11,154. These are distinct quantities.
- DepMap Public 26Q1 shows HDAC3 as the dominant pan-cancer class I HDAC dependency. Genetic-loss context does not identify compound efficacy, selectivity, or safety.

### Docking controls

Docking includes 4LXZ crystallographic redocking, a positive control, an O-methyl zinc-chelation sensitivity decoy, three docking seeds, pose clustering, zinc-geometry reporting, and 4BKX/HDAC1 receptor sensitivity. The decoy illustrates that a favorable docking score can coexist with implausible prespecified zinc-binding geometry. Docking is reported as protocol/receptor sensitivity with unresolved pose plausibility, not proof of binding or HDAC engagement.

### External drug sensitivity

No direct candidate-matched public viability response was available for TC-H-106 in the frozen audit. The Entinostat GDSC surrogate correlation (`R = -0.052`, `P = 0.859`) is a null result and is not used as validation. No new wet-lab assay is claimed.

## Removed or retired analyses

The revision excludes the following from scientific inference:

- simulated TCGA survival and treated/untreated interpretations;
- simulated stage and “stage-independent efficacy” claims;
- hard-coded or unreproducible TMB conclusions;
- non-equivalent DeepCE/ChemCPA proxy comparisons;
- the hard-coded `Pearson R = 0.2841` legacy ablation value;
- direct-target language inferred from reversal genes;
- GDSC surrogate validation claims;
- stable measured-support claims for RG2833 or Tianeptinaline/BG-1010;
- MMFF94 energy as thermodynamic, binding, or efficacy validation.

The full exclusion policy is documented in [`revision/analysis_scope_and_exclusions.md`](revision/analysis_scope_and_exclusions.md).

## Reproducing the corrected post-processing

The GPU training outputs are frozen. Final statistical post-processing and Supplement assets are CPU workflows:

```bash
python scripts/finalize_revision_statistics.py
MPLCONFIGDIR=/tmp/matplotlib-cache python scripts/build_supplement_assets.py
```

Key corrected outputs are under:

- `results/revision/final_statistics/`
- `results/revision/measured_lincs_figures/`
- `results/revision/candidate_stability/`
- `results/revision/reversal_gene_networks_pathways/`
- `results/revision/depmap_hdac_background/`
- `results/revision/docking_controls/`

Run-level parameters, selected checkpoints, hashes, seeds, runtimes, software versions, and evidence limitations are supplied in the machine-readable manifests and revision supplementary tables.

## Submission materials

The synchronized clean manuscript, response letter, Supplement, machine-readable workbook, and LaTeX sources are stored under `revision/submission/`. The reviewer tracker records 17/17 comments as addressed in the revised package.

## License and responsibility

Users must follow the access and reuse terms of TCGA/GDC, LINCS/GEO, DepMap, RCSB PDB, STRING, and g:Profiler. All scientific conclusions should be interpreted within the evidence boundaries above.
