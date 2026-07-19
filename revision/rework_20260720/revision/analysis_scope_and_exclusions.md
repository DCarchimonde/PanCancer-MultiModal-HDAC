# Major-revision analysis scope and exclusions

This file records the evidence freeze used by the revised BMC Bioinformatics submission.

## Primary evidence

- Five leakage-aware model evaluations with repeated seeds.
- Corrected annotation-defined HDAC holdout: 53,839 train profiles and 1,856 test profiles from 30 exact-held-out structures.
- Formal class I HDAC enrichment under signed wTRS, with co-primary Spearman reversal reported as a non-significant metric-sensitivity result.
- Official-condition HiQ LINCS measured reversal and corrected-HDAC prediction-to-measurement association with 10,000 crossed candidate-cancer bootstrap iterations.
- Candidate stability across 48 primary configurations; 72 legacy-inclusive configurations are sensitivity only.
- Frozen candidate tiers, structural-neighbor exposure, reversal-associated networks/pathways, DepMap 26Q1 context, and controlled docking sensitivity.

## Explicitly excluded or downgraded

1. Simulated TCGA survival and treated/untreated interpretation: removed.
2. Simulated stage and stage-independent efficacy: removed.
3. Unsupported or unreproducible legacy TMB claims: removed.
4. Non-equivalent DeepCE/ChemCPA proxies: removed and legacy entry point deprecated.
5. Unsupported cross-attention ablation value not linked to a run artifact: removed and legacy entry point deprecated.
6. Reversed genes as direct drug targets: replaced by reversal-associated genes.
7. Non-significant Entinostat GDSC surrogate as validation: removed from inference; retained only as a null availability record.
8. RG2833 as measured/stable core candidate: downgraded to prediction-only because no measured LINCS signature exists.
9. Tianeptinaline/BG-1010 as an identified core candidate: excluded because of name-structure conflict and exact training exposure.
10. MMFF94 absolute energy as thermodynamic or biological validation: removed and legacy entry point deprecated.
11. Docking score as binding validation: replaced by control-centered protocol/receptor sensitivity.

## Evidence boundary

The two predictors use chemical structure only and do not receive cell identity, dose, or time. Outputs are structure-conditioned central tendencies across training conditions. Measured LINCS reversal is expression evidence, not direct evidence of viability, efficacy, selectivity, target engagement, or clinical benefit.
