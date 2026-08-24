# Leakage-Aware and Auditable Pan-Cancer Transcriptomic Reversal

This repository contains the leakage-aware analysis and auditable evidence package for:

> **A Leakage-Aware and Auditable Framework Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal**

The study is a hypothesis-generating computational framework. Its principal methodological contribution is rigorous data curation, entity-aware splitting, leakage and structural-proximity auditing, measured-profile concordance, and calibrated evidence layers. Molecular-representation comparison is one component of this framework. The study does **not** claim drug efficacy, clinical benefit, direct target discovery, biochemical target engagement, or superiority of the dual-stream model.

## Frozen evidence snapshot

- First-round submission branch: `major-revision-2026`
- Second-round working branch: `second-revision-2026-08`
- Corrected analysis snapshot: `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`
- First-round submission snapshot: `0eb798f` (`round1-submitted-2026-07-25`)
- Perturbation profiles: 55,695 LINCS L1000 Level 5 profiles
- Output genes: 12,328
- Screened canonical compounds: 28,477
- Disease signatures: 22 TCGA cancer types
- Neural hardware recorded by run manifests: NVIDIA GeForce RTX 4090
- CPU model and RAM: not captured in the frozen AutoDL manifests and therefore not inferred

Large public source datasets are not redistributed because of file size and source-specific access terms. The repository contains analysis code and retained, redistributable evidence products. The exact scope of reproducibility is documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); GitHub plus current public inputs alone should not be described as a self-contained, bitwise end-to-end archive.

## What the models estimate

Both predictors receive chemical structure only. They do not receive cell-line identity, dose, exposure time, or other experimental-condition covariates. Their predictions are therefore structure-conditioned central tendencies across training conditions, not condition-specific responses.

The evaluated models are:

- `dual_stream`: atom-token Transformer stream fused with a Morgan-fingerprint stream;
- `fingerprint_mlp`: a strong conventional Morgan-fingerprint multilayer perceptron comparator.

Across pair, leave-drug, leave-cell-line, scaffold, and corrected annotation-defined HDAC holdout evaluations, the dual-stream architecture provided no measurable gain over the fingerprint MLP. The fingerprint comparator was slightly better on average in four settings; performance was comparable in the corrected HDAC holdout. The larger dual-stream model also required longer observed training times. The added representation complexity is therefore not justified by the current benchmarks. Old claims of direct comparison with DeepCE or ChemCPA were invalid because the legacy scripts contained non-equivalent proxy implementations; those entry points now remain only as deprecation notices.

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

Class I HDAC compounds are formally enriched at the fixed top 0.5%, 1%, 5%, and 10% revision cutoffs under signed wTRS after multiplicity correction. For each metric, the ranking equally averages 528 within-library rank fractions per compound (24 screening runs across four generalization regimes, two models, and three seeds, multiplied by 22 cancers); 0 is the strongest rank, so lower is better. Candidate stability uses a separate strength percentile, where 100 is strongest and higher is better. The same class enrichment is not significant under Spearman reversal. Signed wTRS is sensitive to perturbational amplitude, whereas Spearman correlation is invariant to positive rescaling. The result is therefore restricted to **signed wTRS**, may partly reflect response magnitude, and is not a universal HDAC-class effect; the cutoffs were not prospectively preregistered.

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

Measured reversal uses official LINCS condition metadata and distinguishes all-profile, QC-pass, and `is_hiq` strata. Profiles are aggregated within candidate/dose/time/official-cell conditions before candidate--cancer summaries. The analysis supports predicted--measured transcriptomic concordance for the eight evaluated compounds. Without a condition-matched non-HDAC comparator, it does not establish preferential validation relative to alternative perturbagens. It is experimentally measured expression evidence, but not viability, efficacy, binding, or validation on an independent experimental platform.

The prediction--measurement analysis uses a two-way candidate-by-cancer bootstrap (10,000 iterations; seed 20260719) plus leave-one-candidate/cancer sensitivity. A fully condition-matched non-HDAC null could not be recovered from the archived official-condition outputs. An unmatched 11,445-compound all-profile rank analysis is provided only as sensitivity evidence.

### Biological context

- Network and enrichment nodes are **reversal-associated genes**, not direct drug targets.
- The custom g:Profiler request submitted 11,144 unique measured Entrez identifiers; the service reported an effective mapped domain of 11,154. These are distinct quantities. All 5,037 term intersections were rebuilt from the frozen mapped-query order, with GO evidence codes stored separately, and every reconstructed gene count matched the service response.
- DepMap Public 26Q1 shows HDAC3 as the dominant pan-cancer class I HDAC dependency. Genetic-loss context does not identify compound efficacy, selectivity, or safety.

### Docking controls

Docking includes 4LXZ crystallographic redocking, a positive control, an O-methyl zinc-chelation sensitivity decoy, three docking seeds, pose clustering, zinc-geometry reporting, and 4BKX/HDAC1 receptor sensitivity. The decoy illustrates that a favorable docking score can coexist with implausible prespecified zinc-binding geometry. Docking is reported as protocol/receptor sensitivity with unresolved pose plausibility, not proof of binding or HDAC engagement.

### External drug sensitivity

No direct candidate-matched public viability response was available for TC-H-106 in the frozen audit. The Entinostat GDSC surrogate correlation (`R = -0.052`, `P = 0.859`) is a null result and is not used as validation. No new wet-lab assay is claimed.

## Removed or retired analyses

The revision excludes the following from scientific inference:

- simulated TCGA survival and treated/untreated interpretations;
- simulated stage and “stage-independent efficacy” claims;
- unsupported or unreproducible legacy TMB conclusions;
- non-equivalent DeepCE/ChemCPA proxy comparisons;
- the unsupported legacy `Pearson R = 0.2841` ablation value, which was not linked to a reproducible run artifact;
- direct-target language inferred from reversal genes;
- GDSC surrogate validation claims;
- stable measured-support claims for RG2833 or Tianeptinaline/BG-1010;
- MMFF94 energy as thermodynamic, binding, or efficacy validation.

The full exclusion policy is documented in [`revision/analysis_scope_and_exclusions.md`](revision/analysis_scope_and_exclusions.md).

## Reproducibility boundary

The submission documents can be recompiled from the frozen LaTeX, generated table fragments, and vector figure assets:

```bash
cd revision/second_revision_20260824/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main_bmc_bioinformatics_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_bmc_bioinformatics_marked.tex

cd ../supplement
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

Key corrected outputs are under:

- `results/revision/final_statistics/`
- `results/revision/measured_lincs_figures/`
- `results/revision/candidate_stability/`
- `results/revision/reversal_gene_networks_pathways/`
- `results/revision/depmap_hdac_background/`
- `results/revision/docking_controls/`

The complete training and screening code is archived, but upstream analysis scripts require local inputs and run directories that are not all in the portable Git tree. The repository does not contain the large public primary data, GPU checkpoints, prediction arrays, expression caches, or every analysis-time intermediate. Consequently, a bitwise end-to-end rerun from the repository and current public inputs alone is not guaranteed. Run-level parameters, selected checkpoint hashes, seeds, runtimes, software versions, retained outputs, and known missing artifacts are recorded in the machine-readable supplement and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Submission materials

The first-round synchronized package is preserved under `revision/submission/`. Second-round sources and deliverables are kept separately under `revision/second_revision_20260824/`, so the submitted first-round snapshot is not overwritten.

## License and responsibility

Users must follow the access and reuse terms of TCGA/GDC, LINCS/GEO, DepMap, RCSB PDB, STRING, and g:Profiler. All scientific conclusions should be interpreted within the evidence boundaries above.
