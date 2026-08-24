# Second-round LaTeX source build guide

The second-round source tree contains four independently compilable documents:
a clean production manuscript, a reviewer-marked manuscript showing only the
minor-revision changes in dark red, the Supplement, and the second-round
point-by-point response. The first-round response is retained as an archive and
is not part of the second-round upload.

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main_bmc_bioinformatics_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_bmc_bioinformatics_marked.tex

cd ../supplement
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex

cd ../response
latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_reviewers_round2.tex
```

The marked manuscript is a reviewer aid and does not replace the clean
production manuscript. Dark red is restricted to changes made after the
first-round submission snapshot (`0eb798f`). The manuscript and Supplement use
standard TeX Live packages. All PDF figures referenced by the TeX files are
bundled. Generated Supplement table TeX files and their machine-readable CSV
counterparts are under `supplement/generated/`.

`scripts/finalize_revision_statistics.py` performs the final CPU statistical post-processing from the frozen revision results. `scripts/build_supplement_assets.py` regenerates Supplement tables and the six complementary diagnostic figure groups when the repository result tree is available. `scripts/build_revision_workbook.mjs` builds Additional file 2 in the Codex artifact runtime; the already generated `.xlsx` is the submission artifact.

`scripts/enlarge_submission_figure_fonts.py` applies the final vector-only figure-legibility pass to all main and Supplementary PDF figures. It enlarges retained labels without changing plotted data, removes the redundant Figure 2 and Figure 5 footers, keeps Figure 6B row labels outside the heatmap, checks page bounds, and is idempotent.

`scripts/refine_network_figures.py` regenerates main Figure 7 and Supplementary Figure S4 directly from the frozen STRING response caches and frozen candidate-gene bundle. It preserves the audited node/edge subsets while packing disconnected components and placing hub labels with collision checks. Run it after `build_supplement_assets.py`; the generic font-enlargement pass deliberately excludes these two collision-safe figures.

The Supplement figure files are intentionally distinct from the main-manuscript figure files. They report seed-level, condition-level, expanded-stability, expanded-network, lineage-level, and pose-level diagnostics instead of repeating the main panels.

Frozen corrected-evidence commit: `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`.
First-round submission snapshot: `0eb798f` (`round1-submitted-2026-07-25`).
The exact rerun boundary is documented in the repository root
`REPRODUCIBILITY.md` file.
