# Final QA report (internal; do not upload unless requested)

Evidence snapshot: `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`  
Final package QA date: 2026-07-21

## GitHub disconnect recovery

- The branch snapshot present immediately after the interrupted turn contained submission files 01--06 but did not contain the promised 07 internal assessment or 08 complete package.
- The stale aggregate ZIP was not reused. Files 01, 01A, and 02--08 were regenerated from the corrected sources and are synchronized in the final commit.

## Document integrity

- Clean revised manuscript: 36 pages; reviewer-marked manuscript: 36 pages; nine main-figure groups; 36 bibliography entries; no blank page.
- The original uploaded TeX also contained 36 bibliography entries. The revision replaces five obsolete citations with five sources directly required by the corrected workflow; it does not reduce 38 references to 16.
- Response letter: 10 pages; 17/17 reviewer-comment sections; no blank page; every location was refreshed against the final clean-manuscript anchors (scientific text lines 14--611).
- Additional file 1: 30 pages; 24 numbered Supplementary table themes (`S1`--`S24`); six complementary Supplementary figure groups (`S1`--`S6`); no blank page.
- Additional file 2: 58 sheets (README, Sheet Index, Data Dictionary, and 55 data sheets).
- All four PDFs were rendered page by page and visually reviewed as contact sheets plus full-size key pages.
- Every font in all four PDFs is embedded. The two overlay-generated Helvetica instances were explicitly embedded before the final build.
- The 13 main figure assets and six Supplementary figure assets have zero exact SHA-256 overlap. Supplementary figures provide seed-, condition-, legacy-sensitivity-, expanded-network-, lineage-, and pose-level diagnostics instead of reusing main figure files.

## Build, workbook, and container checks

- Clean manuscript, reviewer-marked manuscript, Supplement, and response independently compile from the standalone source tree.
- No LaTeX overfull box, undefined-control-sequence, undefined-reference, citation, or changed-label warning remains. Non-actionable underfull wrapping warnings are retained.
- Source ZIP and XLSX ZIP containers pass integrity tests.
- All 58 workbook sheets produced a nonempty visual preview.
- Workbook search found zero `#REF!`, `#DIV/0!`, `#VALUE!`, `#NAME?`, or `#N/A` errors.
- Workbook search found zero AutoDL or scratch absolute paths.
- Sheet Index row/column counts use integer formatting.
- No unresolved `TODO`, `TBD`, `PLACEHOLDER`, `INSERT HERE`, or `To be replaced` marker occurs in the portable source.

## Scientific consistency checks

- Corrected holdout values consistently use 53,839 train profiles, 1,856 test profiles, and 30 profiled test structures; 39 annotation-defined library structures are distinguished from the 30 with matched LINCS modeling profiles.
- The 1,565-profile/21-structure pre-fix row is labeled provenance only.
- Formal class-enrichment ranking explicitly averages 528 equal run--cancer within-library rank fractions per compound and metric (4 splits × 2 models × 3 seeds × 22 cancers; 0 = strongest). Candidate stability separately uses strength percentiles (100 = strongest). The cutoffs are described as fixed for the revision, not prospectively preregistered.
- Primary stability uses 48 configurations; the 72-configuration legacy-inclusive result is sensitivity only.
- All 5,037 g:Profiler term intersections were reconstructed from mapped-query order; Entrez intersection genes and evidence codes are separate; every count matches the service response.
- g:Profiler values are distinguished as 11,144 submitted unique measured Entrez IDs versus an 11,154 effective service domain.
- S23 accurately states that no wet-lab assay was performed and that the reviewer-suggested public-transcriptomic option was addressed.
- Figure 3 identifies its black lines as OLS visual summaries, reports crossed-bootstrap intervals, and does not claim point-level error bars.
- The model is consistently chemical-structure-only; leave-cell-line is not interpreted as learned condition specificity.
- Model conclusions consistently reject a stable dual-stream superiority claim.
- Candidate tiers and RG2833/Tianeptinaline exclusions are consistent across manuscript, Supplement, workbook, response, and README.
- Measured reversal, networks, DepMap, docking, and GDSC null results use the specified evidence-boundary language.

## Author-locked items

- The original submitted title is retained exactly. It is defensible as a computational-prioritization description because formal signed-wTRS enrichment is reported, while metric dependence and lack of architecture superiority are explicit.
- The AI-assisted technologies declaration is unchanged from the prior package at the author's request.
- A clean production manuscript and a separate dark-red reviewer-marked copy are supplied; the marked copy is not classified as an Additional file.

## Known disclosed limitations, not QA failures

- A condition-matched non-HDAC official-condition null could not be reconstructed from the frozen archive; the unmatched 11,445-compound sensitivity analysis is labeled accordingly.
- The analysis-time 83,490-row condition-level CSV and its SHA256 were not retained in the frozen portable package; the manifest discloses this rather than inventing a checksum.
- Per-cohort final TCGA tumor/normal sample counts were not preserved in the frozen analysis archive and were not guessed. The repeated disease-matrix and observed-mask checksums are explicitly labeled as global rather than cohort-specific hashes.
- CPU model and RAM were not captured by the AutoDL manifests and remain `not recorded`.
- No new wet-lab viability, biochemical, organoid, animal, or clinical experiment was performed.
- Author identities, affiliations, contribution statements, grant identifiers, and policy compliance require final human confirmation.
- Acceptance cannot be guaranteed; this report establishes consistency and integrity of the available evidence, not an editorial outcome.
