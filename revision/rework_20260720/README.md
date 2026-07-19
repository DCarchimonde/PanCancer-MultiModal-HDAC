# LaTeX source build guide

The source archive contains three independently compilable documents:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main_bmc_bioinformatics_revised.tex

cd ../supplement
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex

cd ../response
latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
```

The manuscript and Supplement use standard TeX Live packages. All PDF figures referenced by the TeX files are bundled. Generated Supplement table TeX files and their machine-readable CSV counterparts are under `supplement/generated/`.

`scripts/finalize_revision_statistics.py` performs the final CPU statistical post-processing from the frozen revision results. `scripts/build_supplement_assets.py` regenerates Supplement tables and the six complementary diagnostic figure groups when the repository result tree is available. `scripts/build_revision_workbook.mjs` builds Additional file 2 in the Codex artifact runtime; the already generated `.xlsx` is the submission artifact.

The Supplement figure files are intentionally distinct from the main-manuscript figure files. They report seed-level, condition-level, expanded-stability, expanded-network, lineage-level, and pose-level diagnostics instead of repeating the main panels.

Frozen corrected-evidence commit: `fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f`.
