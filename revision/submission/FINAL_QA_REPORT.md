# Final QA report (internal; do not upload unless requested)

Evidence baseline: `5bdbd60696defe47a682f92b9d03da073234b3a8`
Final package QA date: 2026-07-23

## Final figure-legibility pass

- Typography was enlarged across all 13 main-figure assets and all six Supplementary figure assets, with 1,622 retained text objects increased while preserving the plotted raster/vector data layers.
- Main Figure 1 now uses substantially larger text for all 176 heatmap annotations; the corresponding 176 annotations in Figure 2 and all cell annotations in Supplementary Figure S3 were enlarged as well.
- The redundant bottom methods/evidence-boundary lines in Figures 2 and 5 were removed because the same information is already stated in the Methods and figure captions.
- Figure 6B pathway row labels retain their original right-aligned axis boundary after enlargement, so no label enters the first heatmap column; the longest labels automatically fall back to the largest size that fits the page.
- Automated comparison against the frozen originals confirmed identical displayed numeric-token multisets after excluding the intentionally removed Figure 2 and Figure 5 footers. Every retained text bounding box lies within its PDF page, and final manuscript-scale renders show no overlap or clipping.

## Final shallow correction pass

- Supplementary Table S11 now reports that the seed-paired 95% interval favored the fingerprint baseline for PCI-24781, while explicitly identifying all three-seed intervals as uncertainty summaries rather than population-level significance tests.
- Supplementary Figure S3 uses black annotation text on bright yellow/light-green heatmap cells and white text on dark cells. All 56 high-value annotations were contrast-repaired without changing any displayed value, panel, axis, or caption.
- The reviewer-marked manuscript keeps substantive revisions in dark red while selected minor editorial updates remain black; the clean manuscript content is unchanged apart from the requested acknowledgement.
- The Acknowledgements now thank the handling editor and reviewers for constructive comments that improved the manuscript's clarity and overall presentation.

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
- Every font in all four PDFs is embedded, including the font used for the Figure S3 contrast repair.
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
- A clean production manuscript and a separate reviewer-marked copy are supplied; substantive revisions are dark red and selected minor editorial updates remain black. The marked copy is not classified as an Additional file.

## Known disclosed limitations, not QA failures

- A condition-matched non-HDAC official-condition null could not be reconstructed from the frozen archive; the unmatched 11,445-compound sensitivity analysis is labeled accordingly.
- The analysis-time 83,490-row condition-level CSV and its SHA256 were not retained in the frozen portable package; the manifest discloses this rather than inventing a checksum.
- Per-cohort final TCGA tumor/normal sample counts were not preserved in the frozen analysis archive and were not guessed. The repeated disease-matrix and observed-mask checksums are explicitly labeled as global rather than cohort-specific hashes.
- CPU model and RAM were not captured by the AutoDL manifests and remain `not recorded`.
- No new wet-lab viability, biochemical, organoid, animal, or clinical experiment was performed.
- Author identities, affiliations, contribution statements, grant identifiers, and policy compliance require final human confirmation.
- Acceptance cannot be guaranteed; this report establishes consistency and integrity of the available evidence, not an editorial outcome.
