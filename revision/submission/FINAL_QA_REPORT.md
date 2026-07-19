# Final QA report (internal; do not upload unless requested)

Evidence snapshot: `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`  
QA date: 2026-07-19

## Document integrity

- Revised manuscript: 36 pages; 36 bibliography entries; no blank page.
- Response letter: 10 pages; 17/17 review-comment sections; no blank page.
- Additional file 1: 35 pages; 24 numbered Supplementary table themes (`S1`--`S24`); 6 numbered Supplementary figure groups (`S1`--`S6`); no blank page.
- Additional file 2: 55 sheets (README, index, dictionary, 52 data sheets).
- All three PDFs opened, rendered, and were visually reviewed as page contacts and selected full-size pages.
- All PDF fonts are embedded.

## Build and format checks

- Manuscript, Supplement, and response source compile successfully from the standalone source ZIP.
- No LaTeX overfull box, undefined-control-sequence, undefined-reference, or changed-label warning remains in the final build logs.
- Source ZIP and XLSX ZIP containers pass integrity tests.
- Workbook inspection found no `#REF!`, `#DIV/0!`, `#VALUE!`, or `#NAME?` errors.
- No AutoDL or scratch absolute path occurs in the final workbook or portable source bundle.
- No unresolved `TODO`, `TBD`, `PLACEHOLDER`, `INSERT HERE`, or `To be replaced` marker remains.

## Scientific consistency checks

- Corrected holdout values consistently use 53,839 train profiles, 1,856 test profiles, and 30 test structures; the 1,565/21 pre-fix row is labeled provenance only.
- Primary stability uses 48 configurations; the 72-configuration legacy-inclusive result is sensitivity only.
- g:Profiler values are distinguished as 11,144 submitted IDs versus 11,154 effective mapped domain.
- The model is consistently described as chemical-structure-only and not condition aware.
- Model conclusions consistently reject a stable dual-stream superiority claim.
- Candidate tiers and RG2833/Tianeptinaline exclusions are consistent across manuscript, Supplement, workbook, response, and README.
- Measured reversal, networks, DepMap, docking, and GDSC null results use the specified evidence-boundary language.
- Response figure references were checked against the final manuscript: network/pathway Figure 6, network Figure 7, DepMap Figure 8, and docking Figure 9.

## Known disclosed limitations, not QA failures

- A condition-matched non-HDAC official-condition null could not be reconstructed from the frozen archive; the unmatched 11,445-compound sensitivity analysis is labeled accordingly.
- Per-cohort final TCGA tumor/normal sample counts were not preserved in the frozen analysis archive and were not guessed.
- CPU model and RAM were not captured by the AutoDL manifests and remain `not recorded`.
- No new wet-lab viability, biochemical, organoid, animal, or clinical experiment was performed.
- Acceptance cannot be guaranteed; the report establishes consistency and integrity of the available evidence, not an editorial outcome.
