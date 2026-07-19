#!/usr/bin/env python3
"""Propagate audited revision fixes into the portable submission source tree.

This script is intentionally network-free.  It consumes the frozen g:Profiler
cache repair outputs plus the already exported submission assets, regenerates
the affected Supplement tables, records the omitted condition-table manifest,
and updates two vector-figure labels whose original wording overstated the
evaluation design.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import fitz
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("revision/rework_20260720"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/revision"),
    )
    return parser.parse_args()


def load_builder(source_root: Path):
    path = source_root / "scripts" / "build_supplement_assets.py"
    spec = importlib.util.spec_from_file_location("portable_supplement_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_vector_text(
    path: Path,
    old: str,
    new: str,
    *,
    fontsize: float,
    expand_left: float = 0,
    expand_right: float = 0,
) -> None:
    document = fitz.open(path)
    matches: list[tuple[int, fitz.Rect]] = []
    for page_number, page in enumerate(document):
        for rect in page.search_for(old):
            matches.append((page_number, rect))
    if len(matches) != 1:
        document.close()
        raise RuntimeError(f"Expected one match for {old!r} in {path}; observed {len(matches)}")
    page_number, rect = matches[0]
    page = document[page_number]
    replacement_rect = fitz.Rect(
        max(0, rect.x0 - expand_left),
        rect.y0 - 2,
        min(page.rect.width, rect.x1 + expand_right),
        rect.y1 + 3,
    )
    page.add_redact_annot(replacement_rect, fill=(1, 1, 1))
    page.apply_redactions()
    status = page.insert_textbox(
        replacement_rect,
        new,
        fontsize=fontsize,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_LEFT,
    )
    if status < 0:
        document.close()
        raise RuntimeError(f"Replacement text did not fit in {path}: {new!r}")
    temporary = path.with_suffix(".repaired.pdf")
    document.save(temporary, garbage=4, deflate=True)
    document.close()
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    results_root = args.results_root.resolve()
    generated = source_root / "supplement" / "generated"
    figures = source_root / "manuscript" / "figures"
    generated.mkdir(parents=True, exist_ok=True)

    builder = load_builder(source_root)
    network_dir = results_root / "reversal_gene_networks_pathways"
    full = pd.read_csv(network_dir / "gprofiler_enrichment_full.csv")
    required = {
        "candidate",
        "gene_set_scope",
        "source",
        "native",
        "name",
        "p_value",
        "significant",
        "query_size",
        "term_size",
        "intersection_size",
        "effective_domain_size",
        "precision",
        "recall",
        "intersection_genes",
        "intersection_evidence_codes",
    }
    missing = sorted(required - set(full.columns))
    if missing:
        raise RuntimeError(f"Corrected enrichment table lacks columns: {missing}")
    sanitized = full[list(required)].copy()
    ordered = [
        "candidate",
        "gene_set_scope",
        "source",
        "native",
        "name",
        "p_value",
        "significant",
        "query_size",
        "term_size",
        "intersection_size",
        "effective_domain_size",
        "precision",
        "recall",
        "intersection_genes",
        "intersection_evidence_codes",
    ]
    sanitized = sanitized[ordered]
    sanitized.to_csv(generated / "gprofiler_enrichment_sanitized.csv", index=False)

    significant = full.loc[full["significant"] == True].copy()  # noqa: E712
    scope = (
        significant.groupby(["candidate", "gene_set_scope", "source"], as_index=False)
        .size()
        .rename(columns={"size": "significant_terms_fdr_lt_0_05"})
    )
    scope.to_csv(generated / "pathway_scope_counts.csv", index=False)

    representatives = pd.read_csv(network_dir / "gprofiler_enrichment_representative.csv")
    representatives = representatives.loc[representatives["significant"] == True].copy()  # noqa: E712
    representatives = representatives.sort_values(
        ["candidate", "source", "p_value", "representative_rank"]
    )
    representatives["norm_name"] = (
        representatives["name"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    representatives = (
        representatives.drop_duplicates(["candidate", "source", "norm_name"])
        .groupby(["candidate", "source"])
        .head(2)
    )
    representative_columns = [
        "candidate",
        "gene_set_scope",
        "source",
        "native",
        "name",
        "p_value",
        "intersection_size",
        "term_size",
        "effective_domain_size",
    ]
    representatives = representatives[representative_columns]
    representatives.to_csv(
        generated / "pathway_representative_nonredundant.csv",
        index=False,
    )

    table_s19 = builder.make_table(
        scope,
        ["candidate", "gene_set_scope", "source", "significant_terms_fdr_lt_0_05"],
        ["Candidate", "Gene-set scope", "Source", "Significant terms"],
        ["text", "text", "text", "int"],
        r"Scope of significant g:Profiler results by candidate, reversal direction, and ontology/source.",
        "tab:s_path_scope",
        aligns="lllr",
        font=r"\footnotesize",
        longtable=True,
    )
    table_s19 += builder.make_table(
        representatives,
        representative_columns,
        [
            "Candidate",
            "Gene-set scope",
            "Source",
            "Term ID",
            "Nonredundant term",
            "$P_{FDR}$",
            "Overlap",
            "Term size",
            "Effective domain",
        ],
        ["text", "text", "text", "text", "text", "sci", "int", "int", "int"],
        r"Supplementary Table S19 (continued). Representative nonredundant enrichment terms after corrected gene-intersection redundancy filtering. Probabilities are shown in scientific notation rather than rounded to zero.",
        "tab:s_path_rep",
        aligns=r"p{2.0cm}p{3.1cm}p{1.2cm}p{2.2cm}p{6.3cm}rrrr",
        font=r"\scriptsize",
        landscape=True,
        longtable=True,
        numbered=False,
    )
    (generated / "table_S19.tex").write_text(table_s19, encoding="utf-8")

    audit = json.loads(
        (network_dir / "gprofiler_intersection_repair_audit.json").read_text(encoding="utf-8")
    )
    audit_rows = []
    for record in audit["source_cache_files"]:
        audit_rows.append(
            {
                "record_type": "frozen_response",
                "file": record["file"],
                "candidate": record["candidate"],
                "gene_set_scope": record["gene_set_scope"],
                "sha256": record["sha256"],
                "mapped_query_ids": record["mapped_query_ids"],
                "reported_terms": record["reported_terms"],
                "gprofiler_version": record["gprofiler_version"],
                "timestamp": record["timestamp"],
                "full_terms": audit["full_terms"],
                "representative_terms": audit["representative_terms"],
                "intersection_count_gate": audit["intersection_count_gate"],
                "alignment_source": audit["alignment_source"],
                "evidence_code_policy": audit["evidence_code_policy"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(
        generated / "gprofiler_intersection_repair_audit.csv",
        index=False,
    )
    shutil.copy2(
        network_dir / "gprofiler_intersection_repair_audit.json",
        generated / "gprofiler_intersection_repair_audit.json",
    )

    condition_manifest = pd.DataFrame(
        [
            {
                "repository_relative_path": (
                    "results/revision/measured_lincs_figures/"
                    "condition_level_measured_scores.csv"
                ),
                "rows": 83490,
                "columns": 12,
                "column_names": (
                    "candidate | candidate_role | quality_set | metric | cancer | "
                    "official_cell_iname | time_h | dose_um | condition_median_score | "
                    "signatures | nsample_sum | tas_median"
                ),
                "included_as_formatted_workbook_sheet": False,
                "workbook_representation": (
                    "candidate-cancer, pan-cancer, time, dose, official-condition, "
                    "quality, seed, and crossed-bootstrap aggregates"
                ),
                "nonduplication_reason": (
                    "The full 83,490-row archived table is retained in repository results "
                    "and is not duplicated into the formatted workbook."
                ),
                "source_sha256": "not captured in frozen portable package",
            }
        ]
    )
    condition_manifest.to_csv(generated / "condition_level_manifest.csv", index=False)

    formal_path = generated / "formal_hdac_enrichment_all_metrics.csv"
    formal = pd.read_csv(formal_path)
    formal["ranking_source"] = "cross_generalization_mean_rank_percentile"
    formal["included_splits"] = (
        "leave_drug_out | leave_cell_line_out | scaffold_split | hdac_class_holdout"
    )
    formal["included_models"] = "dual_stream | fingerprint_mlp"
    formal["included_seeds"] = "1 | 2 | 3"
    formal["screening_runs"] = 24
    formal["cancers"] = 22
    formal["run_cancer_observations_per_compound"] = 528
    formal["aggregation_definition"] = (
        "Rank compounds within each run and cancer; convert to library percentile; "
        "average all 528 run-cancer percentiles equally within metric; ascending mean "
        "percentile defines consensus rank."
    )
    formal["cutoff_status"] = "fixed_for_revision_not_prospectively_preregistered"
    formal.to_csv(formal_path, index=False)
    primary_formal = formal.loc[formal["metric"].isin(["signed_wtrs", "spearman_reversal"])]
    table_s16 = builder.make_table(
        primary_formal,
        [
            "metric",
            "annotation_definition",
            "top_fraction",
            "top_k",
            "library_annotated",
            "observed_annotated",
            "expected_annotated",
            "fold_enrichment",
            "fisher_one_sided_p",
            "fdr_bh_within_metric_definition",
        ],
        ["Metric", "Definition", "Top fraction", "Top k", "Library HDAC", "Observed", "Expected", "Fold", "$P$", "FDR"],
        ["text", "text", "pct1", "int", "int", "int", "f2", "f2", "sci", "sci"],
        r"Formal HDAC enrichment at fixed revision library cutoffs. Consensus ranks equally average 24 screening runs across 22 cancers per compound; Fisher tests are one-sided and FDR is within each metric and annotation definition.",
        "tab:s_enrichment",
        aligns="llrrrrrrrr",
        font=r"\scriptsize",
        landscape=True,
        longtable=True,
    )
    (generated / "table_S16.tex").write_text(table_s16, encoding="utf-8")

    external = pd.DataFrame(
        [
            (
                "TC-H-106",
                "Direct public viability response",
                "Unavailable in frozen public-data audit",
                "Not performed",
                "No direct external drug-sensitivity validation claimed",
            ),
            (
                "Entinostat",
                "GDSC surrogate correlation",
                "Available but not candidate matched",
                "Pearson R=-0.052; P=0.859",
                "Null result; cannot validate TC-H-106",
            ),
            (
                "All frozen candidates",
                "New wet-lab viability/biochemical assay",
                "Not performed; reviewer suggested wet-lab assays or public transcriptomic validation",
                "Official-condition measured LINCS used as the public transcriptomic option",
                "Does not replace viability or biochemical validation",
            ),
        ],
        columns=["compound", "endpoint", "availability", "result", "interpretation"],
    )
    external.to_csv(generated / "external_drug_sensitivity_availability.csv", index=False)
    table_s23 = builder.make_table(
        external,
        ["compound", "endpoint", "availability", "result", "interpretation"],
        ["Scope", "Endpoint", "Availability", "Result", "Interpretation"],
        ["text"] * 5,
        r"External direct drug-sensitivity availability and null results. The public-transcriptomic option was completed; no surrogate is presented as direct validation.",
        "tab:s_external",
        aligns=r"p{0.13\textwidth}p{0.17\textwidth}p{0.21\textwidth}p{0.14\textwidth}p{0.22\textwidth}",
        font=r"\footnotesize",
    )
    (generated / "table_S23.tex").write_text(table_s23, encoding="utf-8")

    shutil.copy2(
        network_dir / "reversal_pathway_enrichment_heatmap.pdf",
        figures / "reversal_pathway_enrichment_heatmap.pdf",
    )

    replace_vector_text(
        figures / "screening_landscape_hdac_enrichment.pdf",
        "C  Prespecified top-fraction enrichment",
        "C  Fixed revision top-fraction enrichment",
        fontsize=12,
        expand_right=65,
    )
    stability_figure = source_root / "supplement" / "figures" / "candidate_stability_expanded_72.pdf"
    replace_vector_text(
        stability_figure,
        "C  Recurrent high-rank frequency in OOD regimes",
        "C  Recurrent high-rank frequency across generalization regimes",
        fontsize=10,
        expand_left=45,
        expand_right=60,
    )
    replace_vector_text(
        stability_figure,
        "Fraction of OOD cancer-level observations",
        "Fraction of cancer-level observations across regimes",
        fontsize=9,
        expand_left=20,
        expand_right=45,
    )

    print("PORTABLE SUBMISSION ASSET REPAIR: PASSED")
    print(f"gprofiler_terms={len(full)}")
    print(f"supplement_representative_terms={len(representatives)}")
    print("condition_manifest_rows=1")


if __name__ == "__main__":
    main()
