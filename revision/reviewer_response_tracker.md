# BMC Bioinformatics Major Revision Tracker

Status legend:
- **Evidence ready**: the requested analysis/data audit has been generated; manuscript and response wording remain.
- **Pipeline ready**: code, split, and smoke tests are complete; full 4090 runs remain.
- **Pending**: analysis or manuscript work has not yet been completed.

## Reviewer 1

1. **Dataset composition and transparency** — **Evidence ready**
   - Generated counts for profiles, unique perturbagens, cell lines, dose/time conditions, and replicate numbers.
   - Remaining: add a dataset-composition table and explicit replicate/QC description to Methods/Supplement.

2. **Candidate/HDAC exposure in LINCS and structural analogues** — **Evidence ready (partial)**
   - Exact metadata/SMILES membership audited for TC-H-106, RG2833, and the compound labelled Tianeptinaline.
   - Remaining: nearest-neighbour/Tanimoto analogue analysis and manual identity verification of Tianeptinaline versus BG-1010.

3. **Repeated runs, uncertainty/statistics, and network-pharmacology terminology** — **Pipeline ready (partial)**
   - Deterministic multi-seed training and profile-level metrics are implemented and smoke-tested.
   - Remaining: full runs, confidence intervals/statistical comparison, and replacement of “targets” with “reversal-associated genes”.

## Reviewer 2

4. **Support the statement that pan-cancer ranking consistently prioritised the candidates** — **Pending**
   - Remaining: across-seed/across-split rank stability, top-k frequency, tumour-level consistency, and wording calibrated to the observed stability.

5. **Stronger generalisation evaluation** — **Pipeline ready**
   - Leave-drug-out, leave-cell-line-out, scaffold split, and HDAC-class holdout are generated and verified.
   - Remaining: full dual-stream and fingerprint-baseline training/evaluation.

6. **Remove causal/mechanistic therapeutic implications** — **Pending**
   - Remaining: systematic language audit across Abstract, Results, Discussion, figures, and captions.

7. **Improve figures, captions, font sizes, and readability** — **Pending**

## Reviewer 3

8. **Clarify novelty versus DeepCE, ChemCPA, and related multimodal work** — **Pending**

9. **Repeated random seeds, confidence intervals, and statistical comparison** — **Pipeline ready**
   - Remaining: complete full runs and aggregate results.

10. **Leave-drug-out, leave-cell-line-out, scaffold/external validation** — **Pipeline ready**
   - Internal strict splits are ready; external validation will only be claimed if a suitable independent dataset is actually evaluated.

11. **Minimal biological validation** — **Pending**
   - Current plan: strengthen orthogonal computational evidence and state clearly that no experimental efficacy validation was performed.

12. **Explain prioritisation of TC-H-106 over RG2833/others and add sensitivity analysis** — **Pending**

13. **Strengthen docking controls** — **Pending**
   - Remaining: redocking/co-crystal control and interaction comparison; MD only if feasible and methodologically justified.

14. **Do not present the non-significant GDSC result as supportive validation** — **Pending (text-only, high priority)**

15. **Clarify novelty given established HDAC oncology literature** — **Pending**

16. **Add hyperparameter selection, architecture choice, compute cost, training time, GPU, seeds, normalisation, and uncertainty details** — **Pipeline ready (partial)**
   - Runtime/software/GPU/parameter logging is implemented.
   - Remaining: final values from full runs and manuscript description.

17. **Improve figures and captions** — **Pending**

## Current progress snapshot

- Evidence generated: **2/17**
- Pipelines ready but full 4090 runs pending: **5/17**
- Mostly pending analysis/text/figures: **10/17**
- Formally closed in manuscript + response letter: **0/17**

The 17 comments collapse into six overlapping work packages:
1. Data transparency and candidate exposure
2. Strict generalisation + repeated seeds + statistics
3. Ranking stability and TC-H-106 rationale
4. Novelty/causal-language/GDSC/network-pharmacology revisions
5. Docking and orthogonal biological support
6. Figures, reproducibility details, manuscript revision, and point-by-point response
