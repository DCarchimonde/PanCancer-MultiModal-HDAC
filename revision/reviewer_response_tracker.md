# BMC Bioinformatics major-revision tracker

Evidence freeze: **19 July 2026**  
Submission ID: **8b1e184c-443d-4707-954c-70a65930e16f**

Status legend:

- **Closed**: analysis, manuscript text, supplementary evidence, and point-by-point response are complete.
- **Closed with explicit limitation**: the requested point was addressed using available public evidence, and the remaining unavailable experiment is stated rather than imputed.

## Reviewer 1

1. **Dataset composition and transparency — Closed**
   - 55,695 Level 5 signatures are separated from 11,445 structures, 65 cells, identifiers, times, conditions, and plate/batch-level repeated Level 5 rows.
2. **Candidate/HDAC exposure and structural analogues — Closed**
   - Exact membership, canonical structure, scaffold, and Morgan-neighbor exposure are reported; the corrected HDAC holdout contains 1,856 profiles from 30 structures.
3. **Repeated runs and model comparison — Closed**
   - Thirty-four frozen runs, SDs, paired 95% intervals, and R-squared are reported. No stable dual-stream superiority is claimed.
4. **Network terminology — Closed**
   - All model-derived genes are labelled reversal-associated genes; networks/pathways are complementary annotation, not direct targets or independent validation.

## Reviewer 2

5. **Justify pan-cancer ranking — Closed**
   - Formal prespecified HDAC enrichment is significant for signed wTRS but not co-primary Spearman reversal; the conclusion is metric-dependent.
6. **Stronger generalization — Closed**
   - Pair, leave-drug, leave-cell-line, scaffold, corrected HDAC, strict candidate LDO, and official-condition measured LINCS analyses are reported.
7. **Remove causal/therapeutic implications — Closed**
   - Prediction, measured transcriptomic reversal, biological context, and structural sensitivity are separated from efficacy and target engagement.
8. **Figure quality — Closed**
   - Main and supplementary figures were regenerated as readable vector panels with evidence-boundary captions.

## Reviewer 3

9. **Novelty relative to DeepCE/ChemCPA — Closed**
   - The model is described as an atom-token Transformer plus fingerprint fusion, not a bond-aware GNN or new neural primitive. Non-equivalent proxy comparisons were removed.
10. **Performance and evaluation — Closed**
    - Repeated leakage-aware tests do not support model superiority; all paired intervals are reported.
11. **Minimal biological validation — Closed with explicit limitation**
    - Official measured LINCS transcriptomic profiles were analysed with 10,000 crossed candidate-cancer bootstrap iterations. No viability/biochemical experiment is claimed.
12. **Candidate criteria — Closed**
    - Mocetinostat is core, NCH-51 secondary, TC-H-106 exploratory, RG2833 prediction-only, and Tianeptinaline/BG-1010 identity-conflicted and excluded.
13. **Docking controls — Closed**
    - 4LXZ crystallographic redocking, AutoDock4Zn, a zinc-chelation decoy, controls, three seeds, and 4BKX receptor sensitivity are reported as structural sensitivity.
14. **GDSC null result — Closed**
    - Entinostat surrogate R=-0.052, P=0.859 is retained only as a documented null/unavailable-direct-validation result and is not used as support.
15. **Established HDAC biology — Closed**
    - HDAC oncology relevance is acknowledged as established; brain penetrance is neither inferred nor used as novelty.
16. **Reproducibility — Closed with recorded-data limitation**
    - Architecture, normalization, parameters, seeds, software, GPU, checkpoints, epochs, and runtimes are reported. CPU model and RAM were not captured and are marked not recorded.
17. **Figure captions and biological significance — Closed**
    - Captions explain both biological context and evidential limits; expanded networks and wide panels are in the Supplement.

## Final progress

- Reviewer comments formally closed: **17/17**
- Required experimental packages E1/E2/E3: **complete**
- Conditional direct external drug-sensitivity package C4: **suitable direct public data unavailable; limitation documented**
- Mandatory cleanup package A: **complete in the synchronized revision branch**
- Mandatory manuscript/Supplement/reply package B: **complete and cross-audited**

No new wet-lab viability, biochemical, organoid, animal, or clinical experiment was performed. Measured LINCS support is experimental perturbational transcriptomics, not therapeutic-efficacy validation.
