#!/usr/bin/env python3
"""Build dense, auditable supplementary-table assets from frozen revision outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision"
DERIVED = ROOT / "derived_final"
OUT = ROOT / "supplement" / "generated"
FIG = ROOT / "supplement" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


ROLES = {
    "Mocetinostat": ("core candidate", "Primary computational hypothesis"),
    "NCH-51": ("secondary candidate", "Measured support; scaffold-exposure caveat"),
    "TC-H-106": ("exploratory / original-method-sensitive", "Limited HiQ coverage; close training neighbor"),
    "Belinostat": ("class support", "Measured HDAC-class support"),
    "PCI-24781": ("class support", "Measured HDAC-class support"),
    "Panobinostat": ("class support", "Measured HDAC-class support"),
    "Entinostat": ("reference control", "Established HDAC reference"),
    "Vorinostat": ("reference control", "Established HDAC reference"),
    "RG2833": ("prediction only", "No measured LINCS profile"),
    "Tianeptinaline_or_BG-1010": ("identity conflict; excluded", "Name--structure conflict; exact training exposure"),
}


def save(df: pd.DataFrame, name: str) -> Path:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def latex_escape(value: object) -> str:
    if pd.isna(value):
        return "--"
    text = str(value)
    text = text.replace("Tianeptinaline_or_BG-1010", "Tianeptinaline/BG-1010")
    text = text.replace("Vorinostat_O_methyl_decoy", "Vorinostat O-methyl decoy")
    text = text.replace("designed_zinc_chelation_sensitivity_control", "designed zinc-chelation sensitivity control")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: object, kind: str = "text") -> str:
    if pd.isna(value):
        return "--"
    if kind == "int":
        return f"{int(round(float(value))):,}"
    if kind == "f3":
        return f"{float(value):.3f}"
    if kind == "f2":
        return f"{float(value):.2f}"
    if kind == "f1":
        return f"{float(value):.1f}"
    if kind == "pct1":
        return f"{100 * float(value):.1f}"
    if kind == "sci":
        return f"{float(value):.2e}"
    if kind == "bool":
        return "yes" if bool(value) else "no"
    return latex_escape(value)


def make_table(
    df: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    kinds: list[str],
    caption: str,
    label: str,
    aligns: str | None = None,
    font: str = r"\footnotesize",
    landscape: bool = False,
    longtable: bool = False,
    numbered: bool = True,
) -> str:
    if aligns is None:
        aligns = "l" + "r" * (len(columns) - 1)
    lines: list[str] = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.append(r"\begingroup")
    lines.append(font)
    if longtable:
        caption_line = rf"\caption{{{caption}}}\label{{{label}}}\\" if numbered else rf"\caption*{{{caption}}}\\"
        lines.extend([
            rf"\begin{{longtable}}{{@{{}}{aligns}@{{}}}}",
            caption_line,
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{len(columns)}}}{{l}}{{\textit{{Table continued}}}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{len(columns)}}}{{r}}{{\textit{{Continued on next page}}}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ])
    else:
        caption_lines = [rf"\caption{{{caption}}}", rf"\label{{{label}}}"] if numbered else [rf"\caption*{{{caption}}}"]
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            *caption_lines,
            rf"\begin{{tabular}}{{@{{}}{aligns}@{{}}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ])
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(fmt(row[c], k) for c, k in zip(columns, kinds)) + r" \\")
    if longtable:
        lines.append(r"\end{longtable}")
        if not numbered:
            # caption/longtable advances the table counter even for caption*;
            # restore the thematic S-number so continued parts do not renumber later tables.
            lines.append(r"\addtocounter{table}{-1}")
    else:
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    lines.append(r"\endgroup")
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines) + "\n"


def write_table(number: int, text: str) -> None:
    (OUT / f"table_S{number:02d}.tex").write_text(text, encoding="utf-8")


def build_tabular_assets() -> None:
    dataset = pd.DataFrame([
        ("Level 5 signatures", 55695, "Modeling rows after quality control"),
        ("Signature-ID perturbagen tokens", 13314, "Includes instance suffixes"),
        ("Base perturbagen identifiers", 11527, "Normalized BRD identifiers"),
        ("Unique canonical structures", 11445, "Valid RDKit structures"),
        ("Cell lines", 65, "Across the modeling dataset"),
        ("Perturbagen-token--cell combinations", 39266, "Descriptive combinations"),
        ("Perturbagen-token--cell--time--dose conditions", 52834, "Nominal conditions"),
        ("Additional signatures beyond nominal conditions", 2861, "Plate/batch-level Level 5 entries"),
        ("Expression genes", 12328, "Shared disease/perturbation feature space"),
        ("Screened canonical compounds", 28477, "Frozen screening library; distinct from the modeling structures"),
    ], columns=["item", "count", "interpretation"])
    save(dataset, "dataset_composition")
    write_table(1, make_table(dataset, ["item", "count", "interpretation"],
        ["Item", "Count", "Interpretation"], ["text", "int", "text"],
        r"Composition of the quality-controlled LINCS modeling dataset and frozen screening library. Level 5 signatures are not unique compounds or raw biological replicates.",
        "tab:s_dataset", aligns=r"p{0.30\textwidth}rp{0.58\textwidth}", font=r"\small"))

    settings = pd.DataFrame([
        ("Prediction input", "Canonical chemical structure only; no cell-line, dose, or time covariates"),
        ("Target", "12,328-gene LINCS Level 5 perturbation vector"),
        ("Atom stream", "43-dimensional element identity; 3 Transformer layers; 512 pooled dimensions; no bond adjacency"),
        ("Fingerprint stream", "1,024-bit radius-2 Morgan fingerprint; 512-dimensional projection"),
        ("Dual-stream parameters", "38,682,152"),
        ("Fingerprint MLP parameters", "28,415,016"),
        ("Optimization", "MSE; AdamW; learning rate 2e-4; weight decay 1e-4; batch 128"),
        ("Stopping", "Maximum 30 epochs; patience 5; checkpoint by validation MSE"),
        ("Benchmark seeds", "Pair split 1--5; other splits 1--3; grouping seed 42"),
        ("Primary screening metrics", "signed wTRS and negative Spearman disease--perturbation correlation"),
        ("Stability configurations", "48 primary; 72 only when legacy wTRS is added as sensitivity"),
    ], columns=["item", "specification"])
    save(settings, "model_settings")
    write_table(2, make_table(settings, ["item", "specification"], ["Item", "Frozen specification"],
        ["text", "text"], r"Frozen model, optimization, and interpretation settings.",
        "tab:s_settings", aligns="p{0.25\\textwidth}p{0.70\\textwidth}", font=r"\small"))

    split = pd.read_csv(RESULTS / "structural_neighbor_audit" / "structural_split_inventory.csv")
    split["split"] = split["split"].map({
        "pair_split": "Drug--cell pair", "leave_drug_out": "Leave drug out",
        "leave_cell_line_out": "Leave cell line out", "scaffold_split": "Scaffold split",
        "hdac_class_holdout": "Corrected annotation-defined HDAC",
    })
    split["structure_overlap"] = ["present", "zero", "present", "zero", "zero"]
    split["signature_overlap"] = split["train_test_sig_id_overlap"]
    save(split, "split_audit")
    write_table(3, make_table(split,
        ["split", "train_signatures", "test_signatures", "train_unique_structures", "test_unique_valid_structures", "signature_overlap", "train_test_murcko_scaffold_overlap"],
        ["Split", "Train n", "Test n", "Train struct.", "Test struct.", "Sig. ovlp.", "Scaffold ovlp."],
        ["text", "int", "int", "int", "int", "int", "int"],
        r"Leakage and structural-overlap audit. Exact-structure overlap is zero for leave-drug, scaffold, and corrected annotation-defined HDAC evaluations; the scaffold column is the directly recorded Murcko overlap.",
        "tab:s_split", aligns="lrrrrrr", font=r"\scriptsize"))

    summary = pd.read_csv(DERIVED / "generalization_summary_corrected.csv")
    summary = summary.sort_values(["split", "model"])
    for metric in ["mse", "mae", "r2", "pearson_mean", "spearman_mean"]:
        summary[f"{metric}_display"] = summary.apply(
            lambda r, m=metric: f"{r[m + '_mean']:.3f} ({r[m + '_sd']:.3f})", axis=1)
    save(summary, "model_generalization_summary")
    write_table(4, make_table(summary,
        ["split_label", "model_label", "seeds", "test_profiles", "mse_display", "mae_display", "r2_display", "pearson_mean_display", "spearman_mean_display"],
        ["Split", "Model", "Seeds", "Test", "MSE", "MAE", "$R^2$", "Pearson", "Spearman"],
        ["text"] * 9,
        r"Generalization summary as mean (seed SD). Corrected 1,856-profile HDAC runs replace stale pre-correction summaries.",
        "tab:s_model_summary", aligns="llrrrrrrr", font=r"\scriptsize", landscape=True))

    runs = pd.read_csv(DERIVED / "generalization_all_runs_corrected.csv")
    runs["split"] = runs["split"].map({
        "pair_split": "Pair", "leave_drug_out": "Leave drug", "leave_cell_line_out": "Leave cell",
        "scaffold_split": "Scaffold", "hdac_class_holdout": "Corrected HDAC",
    })
    runs["model"] = runs["model"].map({"dual_stream": "Dual", "fingerprint_mlp": "Fingerprint"})
    save(runs, "model_runs_all_34")
    write_table(5, make_table(runs,
        ["split", "model", "seed", "test_profiles", "best_epoch", "runtime_seconds", "mse", "mae", "r2", "pearson_mean", "spearman_mean"],
        ["Split", "Model", "Seed", "Test", "Epoch", "Runtime s", "MSE", "MAE", "$R^2$", "Pearson", "Spearman"],
        ["text", "text", "int", "int", "int", "f1", "f3", "f3", "f3", "f3", "f3"],
        r"All 34 frozen model runs. Runtime includes training and evaluation on the recorded RTX 4090 environment.",
        "tab:s_runs", aligns="llrrrrrrrrr", font=r"\scriptsize", landscape=True, longtable=True))

    paired = pd.read_csv(DERIVED / "generalization_paired_corrected.csv")
    paired["split"] = paired["split"].map({
        "pair_split": "Pair", "leave_drug_out": "Leave drug", "leave_cell_line_out": "Leave cell",
        "scaffold_split": "Scaffold", "hdac_class_holdout": "Corrected HDAC",
    })
    paired["metric"] = paired["metric"].replace({"pearson_mean": "Pearson", "spearman_mean": "Spearman", "mse": "MSE", "mae": "MAE", "r2": "R2"})
    save(paired, "paired_model_differences")
    write_table(6, make_table(paired,
        ["split", "metric", "seeds", "advantage_dual_mean", "advantage_dual_sd", "advantage_dual_ci95_low", "advantage_dual_ci95_high"],
        ["Split", "Metric", "Seeds", "Dual advantage", "SD", r"95\% low", r"95\% high"],
        ["text", "text", "int", "f3", "f3", "f3", "f3"],
        r"Paired model differences. Positive values favor the dual stream; all intervals include zero except the candidate-specific PCI-24781 comparison reported separately.",
        "tab:s_paired", aligns="llrrrrr", font=r"\footnotesize", longtable=True))

    identity = pd.read_csv(RESULTS / "lincs_candidate_validation" / "candidate_identity_map.csv")
    identity["role"] = identity["candidate"].map(lambda x: ROLES.get(x, ("", ""))[0])
    identity["evidence_note"] = identity["candidate"].map(lambda x: ROLES.get(x, ("", ""))[1])
    identity["candidate_display"] = identity["candidate"].replace({"Tianeptinaline_or_BG-1010": "Tianeptinaline / BG-1010"})
    save(identity, "candidate_identity_and_roles")
    write_table(7, make_table(identity,
        ["candidate_display", "drug_ids", "role", "is_hdac_annotated", "name_conflict", "evidence_note"],
        ["Candidate", "BRD ID", "Frozen role", "HDAC annotated", "Name conflict", "Interpretation"],
        ["text", "text", "text", "bool", "bool", "text"],
        r"Candidate identity, BRD identifier, and frozen evidence role. Tianeptinaline/BG-1010 is retained only to document the identity conflict.",
        "tab:s_identity", aligns=r"p{2.8cm}p{2.8cm}p{4.2cm}ccp{7.5cm}", font=r"\scriptsize", landscape=True))

    coverage = pd.read_csv(RESULTS / "measured_lincs_figures" / "candidate_condition_quality_coverage.csv")
    hiq = pd.read_csv(RESULTS / "measured_lincs_figures" / "candidate_pan_cancer_measured_summary_official.csv")
    hiq = hiq[(hiq["quality_set"] == "hiq") & (hiq["metric"] == "signed_wtrs")][["candidate", "official_cell_lines"]]
    hiq = hiq.rename(columns={"official_cell_lines": "hiq_official_cell_lines"})
    coverage = coverage.merge(hiq, on="candidate", how="left")
    save(coverage, "official_condition_coverage")
    write_table(8, make_table(coverage,
        ["candidate", "all_profiles", "qc_pass_profiles", "hiq_profiles", "official_cell_lines", "hiq_official_cell_lines", "times_h", "dose_levels", "dose_um_min", "dose_um_max"],
        ["Candidate", "All", "QC", "HiQ", "All cells", "HiQ cells", "Times h", "Dose levels", r"Min $\mu$M", r"Max $\mu$M"],
        ["text", "int", "int", "int", "int", "int", "text", "int", "f3", "f3"],
        r"Official LINCS condition and quality coverage. ``All cells'' and ``HiQ cells'' are deliberately separated.",
        "tab:s_coverage", aligns="lrrrrrlrrr", font=r"\scriptsize", landscape=True))

    prepost = pd.read_csv(DERIVED / "corrected_hdac_holdout_pre_post_fix.csv")
    save(prepost, "corrected_hdac_pre_post")
    write_table(9, make_table(prepost,
        ["version", "test_signatures", "test_structures", "test_cache_cell_lines", "signature_change_vs_pre_fix", "structure_change_vs_pre_fix"],
        ["Version", "Test profiles", "Test structures", "Test cells", r"$\Delta$ profiles", r"$\Delta$ structures"],
        ["text", "int", "int", "int", "int", "int"],
        r"Pre-correction versus corrected annotation-defined HDAC holdout. The pre-correction row is provenance only and is excluded from inference.",
        "tab:s_prepost", aligns="lrrrrr", font=r"\small"))

    hdac30 = pd.read_csv(DERIVED / "corrected_hdac_holdout_30_structures.csv")
    save(hdac30, "corrected_hdac_30_structures")
    write_table(10, make_table(hdac30,
        ["brd_id", "display_name", "test_signatures", "test_cache_cell_lines", "hdac_annotation_source", "added_by_annotation_fix", "nch51_removed_from_training"],
        ["BRD ID", "Compound", "Profiles", "Cells", "Annotation source", "Added by fix", "NCH-51 removed"],
        ["text", "text", "int", "int", "text", "bool", "bool"],
        r"Complete 30-structure corrected annotation-defined HDAC holdout inventory.",
        "tab:s_hdac30", aligns="lllrrlll", font=r"\footnotesize", longtable=True))

    membership = pd.read_csv(RESULTS / "leave_drug_candidate_validation" / "candidate_leave_drug_membership.csv")
    ldo = pd.read_csv(RESULTS / "leave_drug_candidate_validation" / "leave_drug_candidate_metrics_summary.csv")
    save(membership, "leave_drug_membership")
    save(ldo, "leave_drug_candidate_summary")
    text = make_table(membership,
        ["candidate", "train_signatures", "test_signatures", "test_cell_lines", "strict_leave_drug_out"],
        ["Candidate", "Train", "Test", "Test cells", "Strict LDO"],
        ["text", "int", "int", "int", "bool"],
        r"Strict leave-drug candidate membership.", "tab:s_ldo_member", aligns="lrrrr", font=r"\small")
    text += make_table(ldo,
        ["candidate", "model", "seeds", "signatures", "cell_lines", "pearson_mean", "pearson_ci95_low", "pearson_ci95_high", "spearman_mean"],
        ["Candidate", "Model", "Seeds", "Profiles", "Cells", "Pearson", r"95\% low", r"95\% high", "Spearman"],
        ["text", "text", "int", "int", "int", "f3", "f3", "f3", "f3"],
        r"Supplementary Table S11 (continued). Strict leave-drug candidate metrics. Only Mocetinostat and PCI-24781 were true test candidates.",
        "tab:s_ldo_metrics", aligns="llrrrrrrr", font=r"\scriptsize", numbered=False)
    write_table(11, text)

    twoway = pd.read_csv(DERIVED / "predicted_measured_two_way_bootstrap.csv")
    save(twoway, "predicted_measured_two_way_bootstrap")
    write_table(12, make_table(twoway,
        ["metric", "model", "candidates", "cancers", "pearson", "pearson_two_way_ci95_low", "pearson_two_way_ci95_high", "spearman", "spearman_two_way_ci95_low", "spearman_two_way_ci95_high"],
        ["Metric", "Model", "Candidates", "Cancers", "Pearson", r"95\% low", r"95\% high", "Spearman", r"95\% low", r"95\% high"],
        ["text", "text", "int", "int", "f3", "f3", "f3", "f3", "f3", "f3"],
        r"Corrected-HDAC predicted versus HiQ measured reversal with 10,000 candidate--cancer crossed bootstrap iterations (seed 20260719).",
        "tab:s_twoway", aligns="llrrrrrrrr", font=r"\scriptsize", landscape=True))

    measured = pd.read_csv(RESULTS / "measured_lincs_figures" / "candidate_pan_cancer_measured_summary_official.csv")
    measured = measured[(measured["quality_set"] == "hiq") & measured["metric"].isin(["signed_wtrs", "spearman_reversal"])]
    measured = measured[measured["candidate"] != "Tianeptinaline_or_BG-1010"]
    measured = measured.sort_values(["candidate", "metric"])
    save(measured, "measured_reversal_hiq_primary")
    write_table(13, make_table(measured,
        ["candidate", "metric", "signatures", "official_cell_lines", "median_across_cancers", "cancer_score_q25", "cancer_score_q75", "positive_cancer_fraction"],
        ["Candidate", "Metric", "HiQ", "HiQ cells", "Median", "Q1", "Q3", r"Positive cancers \%"],
        ["text", "text", "int", "int", "f3", "f3", "f3", "pct1"],
        r"Primary official-condition HiQ measured reversal across 22 cancer signatures.",
        "tab:s_measured", aligns="llrrrrrr", font=r"\footnotesize", longtable=True))

    stability = pd.read_csv(DERIVED / "candidate_stability_primary_vs_sensitivity.csv")
    save(stability, "candidate_stability_primary_vs_sensitivity")
    write_table(14, make_table(stability,
        ["candidate", "candidate_role", "run_configurations_primary48", "median_run_library_percentile_primary48", "worst_split_metric_median_percentile_primary48", "run_configurations_sensitivity72", "median_run_library_percentile_sensitivity72", "median_percentile_change_primary_minus_sensitivity"],
        ["Candidate", "Role", "Primary n", "Primary median", "Primary worst", "Sensitivity n", "Sensitivity median", r"$\Delta$ primary--sens."],
        ["text", "text", "int", "f1", "f1", "int", "f1", "f1"],
        r"Candidate stability under the 48-configuration primary hierarchy and 72-configuration legacy-inclusive sensitivity analysis.",
        "tab:s_stability", aligns="llrrrrrr", font=r"\scriptsize", landscape=True))

    neighbor = pd.read_csv(RESULTS / "structural_neighbor_audit" / "candidate_train_similarity_summary.csv")
    neighbor = neighbor[neighbor["split"] == "hdac_class_holdout"].copy()
    neighbor["role"] = neighbor["candidate"].map(lambda x: ROLES.get(x, ("", ""))[0])
    save(neighbor, "candidate_neighbor_corrected_hdac")
    write_table(15, make_table(neighbor,
        ["candidate", "role", "train_signatures", "test_signatures", "max_tanimoto", "mean_top10_tanimoto", "exact_structure_in_train", "candidate_scaffold_in_train", "same_scaffold_training_structures", "name_conflict"],
        ["Candidate", "Role", "Train", "Test", "Max Tanimoto", "Top-10 mean", "Exact train", "Scaffold train", "Same-scaffold n", "Conflict"],
        ["text", "text", "int", "int", "f3", "f3", "bool", "bool", "int", "bool"],
        r"Corrected annotation-defined HDAC structural-neighbor and scaffold exposure audit.",
        "tab:s_neighbor", aligns=r"p{3.2cm}p{3.7cm}rrrrrrrr", font=r"\tiny", landscape=True))

    enrichment = pd.read_csv(DERIVED / "formal_hdac_enrichment.csv")
    enrichment_primary = enrichment[enrichment["metric"].isin(["signed_wtrs", "spearman_reversal"])].copy()
    save(enrichment, "formal_hdac_enrichment_all_metrics")
    write_table(16, make_table(enrichment_primary,
        ["metric", "annotation_definition", "top_fraction", "top_k", "library_annotated", "observed_annotated", "expected_annotated", "fold_enrichment", "fisher_one_sided_p", "fdr_bh_within_metric_definition"],
        ["Metric", "Definition", "Top fraction", "Top k", "Library HDAC", "Observed", "Expected", "Fold", "$P$", "FDR"],
        ["text", "text", "pct1", "int", "int", "int", "f2", "f2", "sci", "sci"],
        r"Formal HDAC enrichment at prespecified library cutoffs. Fisher tests are one-sided; FDR is within each metric and annotation definition.",
        "tab:s_enrichment", aligns="llrrrrrrrr", font=r"\scriptsize", landscape=True, longtable=True))

    threshold = pd.read_csv(RESULTS / "frozen_candidate_biology" / "consensus_threshold_sensitivity.csv")
    threshold70 = threshold[threshold["consensus_threshold"] == 0.7].copy()
    threshold70["selected_top_genes"] = 200
    mapping = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "gene_id_mapping_coverage.csv")
    map200 = mapping[["candidate", "selected_genes", "string_mappable_genes", "mapping_fraction"]]
    threshold70 = threshold70.merge(map200, on="candidate", how="left")
    save(threshold70, "reversal_gene_definition_and_mapping")
    write_table(17, make_table(threshold70,
        ["candidate", "candidate_role", "consensus_threshold", "eligible_genes", "stable_direction_genes", "selected_top_genes", "string_mappable_genes", "mapping_fraction"],
        ["Candidate", "Role", "Threshold", "Eligible", "Stable direction", "Top selected", "STRING mappable", r"Mapping \%"],
        ["text", "text", "pct1", "int", "int", "int", "int", "pct1"],
        r"Frozen reversal-associated gene definition and mapping coverage. Selection precedes network and pathway interpretation.",
        "tab:s_genedef", aligns="llrrrrrr", font=r"\footnotesize", landscape=True))

    nodes = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "string_network_nodes.csv")
    nodes_sanitized = nodes[[
        "candidate", "network_type", "preferred_name", "gene_id", "gene_symbol",
        "degree", "weighted_degree", "component_id", "component_size", "consensus_rank",
        "disease_direction", "predicted_drug_direction", "measured_drug_direction",
        "reversal_pattern", "candidate_role", "biology_scope",
    ]].copy()
    save(nodes_sanitized, "string_nodes_sanitized")
    top = nodes[(nodes["network_type"] == "physical") & nodes["selected_network_gene"]].copy()
    top = top.sort_values(["candidate", "consensus_rank"]).groupby("candidate").head(20)
    top["display_gene"] = top["gene_symbol"].fillna(top["preferred_name"])
    top["rank"] = top.groupby("candidate").cumcount() + 1
    pivot = top.pivot(index="rank", columns="candidate", values="display_gene").reset_index()
    candidate_order = ["Mocetinostat", "NCH-51", "TC-H-106", "Belinostat", "Entinostat", "Vorinostat"]
    pivot = pivot[["rank"] + candidate_order]
    save(pivot, "top20_reversal_associated_genes")
    write_table(18, make_table(pivot,
        ["rank"] + candidate_order,
        ["Rank"] + candidate_order,
        ["int"] + ["text"] * len(candidate_order),
        r"Leading 20 reversal-associated genes per candidate or control by frozen consensus rank. These are not direct drug targets.",
        "tab:s_topgenes", aligns="rllllll", font=r"\footnotesize", landscape=True))

    full = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "gprofiler_enrichment_full.csv")
    enrichment_sanitized = full[[
        "candidate", "gene_set_scope", "source", "native", "name", "p_value",
        "significant", "query_size", "term_size", "intersection_size",
        "effective_domain_size", "precision", "recall", "intersection_genes",
    ]].copy()
    save(enrichment_sanitized, "gprofiler_enrichment_sanitized")
    sig = full[full["significant"] == True].copy()  # noqa: E712
    scope = sig.groupby(["candidate", "gene_set_scope", "source"], as_index=False).size().rename(columns={"size": "significant_terms_fdr_lt_0_05"})
    save(scope, "pathway_scope_counts")
    reps = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "gprofiler_enrichment_representative.csv")
    reps = reps[reps["significant"] == True].sort_values(["candidate", "source", "p_value", "representative_rank"])  # noqa: E712
    reps["norm_name"] = reps["name"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    reps = reps.drop_duplicates(["candidate", "source", "norm_name"]).groupby(["candidate", "source"]).head(2)
    reps = reps[["candidate", "gene_set_scope", "source", "native", "name", "p_value", "intersection_size", "term_size", "effective_domain_size"]]
    save(reps, "pathway_representative_nonredundant")
    text = make_table(scope,
        ["candidate", "gene_set_scope", "source", "significant_terms_fdr_lt_0_05"],
        ["Candidate", "Gene-set scope", "Source", "Significant terms"], ["text", "text", "text", "int"],
        r"Scope of significant g:Profiler results by candidate, reversal direction, and ontology/source.",
        "tab:s_path_scope", aligns="lllr", font=r"\footnotesize", longtable=True)
    text += make_table(reps,
        ["candidate", "gene_set_scope", "source", "native", "name", "p_value", "intersection_size", "term_size", "effective_domain_size"],
        ["Candidate", "Gene-set scope", "Source", "Term ID", "Nonredundant term", "$P_{FDR}$", "Overlap", "Term size", "Effective domain"],
        ["text", "text", "text", "text", "text", "sci", "int", "int", "int"],
        r"Supplementary Table S19 (continued). Representative nonredundant enrichment terms. Probabilities are shown in scientific notation rather than rounded to zero.",
        "tab:s_path_rep", aligns=r"p{2.0cm}p{3.1cm}p{1.2cm}p{2.2cm}p{6.3cm}rrrr", font=r"\scriptsize", landscape=True, longtable=True, numbered=False)
    write_table(19, text)

    net = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "string_network_summary.csv")
    overlap = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "primary_control_gene_overlap.csv")
    save(net, "string_network_summary")
    save(overlap, "primary_control_gene_overlap")
    text = make_table(net,
        ["candidate", "network_type", "selected_genes", "nodes_with_edges", "network_edges", "density", "components", "largest_component_nodes"],
        ["Candidate", "Network", "Selected", "Nodes with edges", "Edges", "Density", "Components", "Largest component"],
        ["text", "text", "int", "int", "int", "f3", "int", "int"],
        r"STRING physical and functional network scope at required score 700.",
        "tab:s_network", aligns="llrrrrrr", font=r"\footnotesize", longtable=True)
    text += make_table(overlap,
        ["candidate_a", "candidate_b", "genes_a", "genes_b", "shared_genes", "jaccard"],
        ["Candidate", "Reference", "Genes A", "Genes B", "Shared", "Jaccard"],
        ["text", "text", "int", "int", "int", "f3"],
        r"Supplementary Table S20 (continued). Primary-candidate overlap with Entinostat and Vorinostat reversal-associated gene sets.",
        "tab:s_overlap", aligns="llrrrr", font=r"\small", numbered=False)
    write_table(20, text)

    dep = pd.read_csv(RESULTS / "depmap_hdac_background" / "analysis" / "depmap_hdac_pan_cancer_summary.csv")
    het = pd.read_csv(RESULTS / "depmap_hdac_background" / "analysis" / "depmap_hdac_lineage_heterogeneity.csv")
    save(dep, "depmap_pan_cancer_summary")
    save(het, "depmap_lineage_heterogeneity")
    text = make_table(dep,
        ["gene", "models", "median_gene_effect", "median_ci95_low", "median_ci95_high", "fraction_gene_effect_le_neg_0_5", "fraction_gene_effect_le_neg_1_0"],
        ["Gene", "Models", "Median", r"95\% low", r"95\% high", r"$\leq-0.5$ \%", r"$\leq-1.0$ \%"],
        ["text", "int", "f3", "f3", "f3", "pct1", "pct1"],
        r"DepMap Public 26Q1 pan-cancer class I HDAC gene-effect summary.",
        "tab:s_depmap", aligns="lrrrrrr", font=r"\small")
    text += make_table(het,
        ["gene", "eligible_lineages", "models", "kruskal_wallis_h", "p_value", "epsilon_squared", "fdr_bh"],
        ["Gene", "Lineages", "Models", "Kruskal $H$", "$P$", r"$\epsilon^2$", "FDR"],
        ["text", "int", "int", "f2", "sci", "f3", "sci"],
        r"Supplementary Table S21 (continued). Lineage heterogeneity. Effect sizes are modest despite FDR significance.",
        "tab:s_depmap_het", aligns="lrrrrrr", font=r"\small", numbered=False)
    write_table(21, text)

    redock = pd.read_csv(RESULTS / "docking_controls" / "redocking_4lxz" / "redocking_method_summary.csv")
    panel = pd.read_csv(RESULTS / "docking_controls" / "controlled_panel_4lxz" / "controlled_docking_summary.csv")
    sens = pd.read_csv(RESULTS / "docking_controls" / "receptor_sensitivity_4bkx" / "cross_receptor_comparison.csv")
    panel["input_sha256_status"] = panel["input_sha256"].apply(lambda x: "not applicable: crystallographic ligand reused" if pd.isna(x) or x == "" else str(x))
    panel["medoid_file"] = panel["medoid_sdf"].apply(lambda x: Path(str(x)).name if not pd.isna(x) else "")
    panel = panel.drop(columns=["medoid_sdf"])
    panel_4bkx = pd.read_csv(RESULTS / "docking_controls" / "receptor_sensitivity_4bkx" / "docking_summary_4bkx.csv")
    panel_4bkx["medoid_file"] = panel_4bkx["medoid_sdf"].apply(lambda x: Path(str(x)).name if not pd.isna(x) else "")
    panel_4bkx = panel_4bkx.drop(columns=["medoid_sdf"])
    save(redock, "redocking_summary")
    save(panel, "docking_4lxz_sanitized")
    save(panel_4bkx, "docking_4bkx_sanitized")
    save(sens, "docking_receptor_sensitivity")
    text = make_table(redock,
        ["method", "seeds", "selected_cluster_seed_coverage", "selected_cluster_pose_count", "selected_cluster_median_score_kcal_mol", "consensus_medoid_rmsd_a", "consensus_medoid_zn_min_a", "consensus_medoid_zn_max_a"],
        ["Method", "Seeds", "Seed coverage", "Cluster poses", "Median score", "Medoid RMSD", "Zn min", "Zn max"],
        ["text", "int", "int", "int", "f3", "f3", "f3", "f3"],
        r"4LXZ crystallographic redocking control. The prespecified RMSD gate was 2 Angstrom.",
        "tab:s_redock", aligns="lrrrrrrr", font=r"\footnotesize")
    text += make_table(panel,
        ["compound", "role", "stable_all_seed_consensus", "selected_cluster_pose_count", "selected_cluster_median_score_kcal_mol", "consensus_medoid_score_kcal_mol", "zbg_min_zn_a", "zbg_max_zn_a", "nearest_heteroatom_zn_a"],
        ["Compound", "Role", "3-seed", "Poses", "Cluster score", "Medoid score", "ZBG min", "ZBG max", "Nearest heteroatom"],
        ["text", "text", "bool", "int", "f3", "f3", "f3", "f3", "f3"],
        r"Supplementary Table S22 (continued). Frozen 4LXZ control/candidate panel. Favorable score is not evidence of intended zinc geometry.",
        "tab:s_panel", aligns=r"p{3.2cm}p{6.0cm}rrrrrrr", font=r"\scriptsize", landscape=True, numbered=False)
    text += make_table(sens,
        ["compound", "fixed_frame_medoid_rmsd_a", "cluster_median_score_4lxz_kcal_mol", "cluster_median_score_4bkx_kcal_mol", "score_delta_4bkx_minus_4lxz_kcal_mol", "nearest_heteroatom_zn_4lxz_a", "nearest_heteroatom_zn_4bkx_a"],
        ["Compound", "Pose RMSD", "4LXZ score", "4BKX score", r"$\Delta$ score", "4LXZ nearest Zn", "4BKX nearest Zn"],
        ["text", "f3", "f3", "f3", "f3", "f3", "f3"],
        r"Supplementary Table S22 (continued). HDAC2 4LXZ versus aligned HDAC1 4BKX receptor sensitivity.",
        "tab:s_receptor", aligns="lrrrrrr", font=r"\footnotesize", numbered=False)
    write_table(22, text)

    external = pd.DataFrame([
        ("TC-H-106", "Direct public viability response", "Unavailable in frozen public-data audit", "Not performed", "No direct external drug-sensitivity validation claimed"),
        ("Entinostat", "GDSC surrogate correlation", "Available but not candidate matched", "Pearson R=-0.052; P=0.859", "Null result; cannot validate TC-H-106"),
        ("All frozen candidates", "New wet-lab viability/biochemical assay", "Not requested in original reviews and not performed", "Unavailable", "Explicit limitation and future-work requirement"),
    ], columns=["compound", "endpoint", "availability", "result", "interpretation"])
    save(external, "external_drug_sensitivity_availability")
    write_table(23, make_table(external,
        ["compound", "endpoint", "availability", "result", "interpretation"],
        ["Scope", "Endpoint", "Availability", "Result", "Interpretation"],
        ["text"] * 5,
        r"External direct drug-sensitivity availability and null results. A surrogate is not presented as validation.",
        "tab:s_external", aligns="p{0.13\\textwidth}p{0.17\\textwidth}p{0.21\\textwidth}p{0.14\\textwidth}p{0.22\\textwidth}", font=r"\footnotesize"))

    repro = pd.DataFrame([
        ("Neural hardware", "NVIDIA GeForce RTX 4090", "Recorded by run manifests"),
        ("CPU model", "Not captured in frozen AutoDL manifests", "Not reconstructed or fabricated"),
        ("RAM", "Not captured in frozen AutoDL manifests", "Not reconstructed or fabricated"),
        ("Python", "3.12.3", "Run manifests"),
        ("PyTorch/CUDA", "2.5.1+cu124", "Run manifests"),
        ("RDKit", "2025.03.6 for docking; modeling canonicalization version in repository environment", "Environment manifests"),
        ("Neural benchmark runtime", "5,112.6 s summed across 34 reported runs", "Run-level values in Table S5"),
        ("Docking runtimes", "330.0 s redocking; 2,121.9 s 4LXZ panel; 2,134.9 s 4BKX sensitivity", "CPU workflows"),
        ("Split seed", "42", "Top-level grouping"),
        ("Run seeds", "1--5 pair; 1--3 other splits", "Frozen benchmark"),
        ("Crossed-bootstrap seed", "20260719", "10,000 iterations"),
        ("Repository evidence snapshot", "fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f", "Corrected statistics, scripts, exclusions, and tracker on major-revision-2026"),
    ], columns=["item", "value", "provenance"])
    save(repro, "reproducibility_inventory")
    write_table(24, make_table(repro,
        ["item", "value", "provenance"], ["Item", "Frozen value", "Provenance/limitation"],
        ["text", "text", "text"], r"Reproducibility inventory. Unrecorded CPU/RAM values are disclosed rather than guessed.",
        "tab:s_repro", aligns="p{0.22\\textwidth}p{0.43\\textwidth}p{0.30\\textwidth}", font=r"\footnotesize"))


def spring_positions(nodes: list[str], edges: list[tuple[str, str]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(nodes)
    pos = rng.normal(0, 0.6, size=(n, 2))
    index = {name: i for i, name in enumerate(nodes)}
    pairs = [(index[a], index[b]) for a, b in edges if a in index and b in index and a != b]
    k = 1.5 / math.sqrt(max(n, 1))
    temperature = 0.15
    for iteration in range(350):
        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(delta * delta, axis=2) + np.eye(n)
        dist = np.sqrt(dist2)
        rep = (k * k / dist2)[:, :, None] * (delta / dist[:, :, None])
        disp = rep.sum(axis=1)
        for i, j in pairs:
            dvec = pos[i] - pos[j]
            d = max(float(np.linalg.norm(dvec)), 1e-3)
            force = (d * d / k) * (dvec / d)
            disp[i] -= force
            disp[j] += force
        disp -= 0.04 * pos
        norms = np.linalg.norm(disp, axis=1)
        step = disp / np.maximum(norms[:, None], 1e-9) * np.minimum(norms, temperature)[:, None]
        pos += step
        temperature *= 0.992
    pos -= pos.mean(axis=0)
    scale = np.max(np.abs(pos))
    if scale > 0:
        pos /= scale
    return {name: pos[index[name]] for name in nodes}


def build_network_figure() -> None:
    nodes_df = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "string_network_nodes.csv")
    edges_df = pd.read_csv(RESULTS / "reversal_gene_networks_pathways" / "string_network_edges.csv")
    candidates = ["Mocetinostat", "NCH-51", "TC-H-106", "Belinostat", "Entinostat", "Vorinostat"]
    inventory = []
    pdf_path = FIG / "supplementary_expanded_physical_networks.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_start in (0, 3):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = axes.flatten()
            for ax_i, candidate in enumerate(candidates[page_start:page_start + 3]):
                ax = axes[ax_i]
                sub_nodes = nodes_df[(nodes_df["candidate"] == candidate) & (nodes_df["network_type"] == "physical")]
                sub_nodes = sub_nodes[sub_nodes["degree"] > 0].sort_values(["weighted_degree", "consensus_rank"], ascending=[False, True]).head(45)
                selected = set(sub_nodes["preferred_name"].astype(str))
                sub_edges = edges_df[(edges_df["candidate"] == candidate) & (edges_df["network_type"] == "physical")].copy()
                sub_edges = sub_edges[sub_edges["preferredName_A"].isin(selected) & sub_edges["preferredName_B"].isin(selected)]
                names = sorted(selected)
                pairs = list(zip(sub_edges["preferredName_A"], sub_edges["preferredName_B"]))
                pos = spring_positions(names, pairs, seed=20260719 + page_start + ax_i)
                node_meta = sub_nodes.set_index("preferred_name")
                for _, e in sub_edges.iterrows():
                    p1, p2 = pos[str(e["preferredName_A"])], pos[str(e["preferredName_B"])]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#AAB3C2", alpha=0.50,
                            lw=0.35 + 1.2 * float(e["score"]))
                top_labels = set(sub_nodes.head(14)["preferred_name"].astype(str))
                for name in names:
                    p = pos[name]
                    meta = node_meta.loc[name]
                    direction = str(meta.get("disease_direction", ""))
                    color = "#D95F59" if direction == "up" else "#4C78A8"
                    size = 20 + 16 * float(meta.get("weighted_degree", 0))
                    ax.scatter(p[0], p[1], s=size, c=color, edgecolors="white", linewidths=0.45, zorder=3)
                    if name in top_labels:
                        ax.text(p[0] + 0.018, p[1] + 0.018, name, fontsize=6.3, weight="semibold")
                ax.set_title(f"{candidate}: {len(names)} non-isolated nodes, {len(sub_edges)} displayed edges", fontsize=10, weight="bold")
                ax.set_xlim(-1.12, 1.15)
                ax.set_ylim(-1.12, 1.12)
                ax.axis("off")
                inventory.append({"candidate": candidate, "displayed_nodes": len(names), "displayed_edges": len(sub_edges),
                                  "selection": "up to 45 non-isolated physical-network nodes by weighted degree"})
            axes[3].axis("off")
            axes[3].text(0.02, 0.84, "Interpretation key", fontsize=13, weight="bold")
            axes[3].text(0.02, 0.70, "Red: disease-up / drug-down reversal-associated gene", fontsize=9, color="#A33C38")
            axes[3].text(0.02, 0.60, "Blue: disease-down / drug-up reversal-associated gene", fontsize=9, color="#365D86")
            axes[3].text(0.02, 0.44, "Node size: physical-network weighted degree\nEdge width: STRING physical score\nLabels: leading hubs only", fontsize=9)
            axes[3].text(0.02, 0.20, "Displayed genes are expanded plotting subsets.\nFull 200-gene sets and all physical/functional edges\nare supplied in Additional file 2.", fontsize=9, style="italic")
            fig.suptitle("Expanded high-confidence STRING physical networks of frozen candidates and controls",
                         fontsize=14, weight="bold", y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=1.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    save(pd.DataFrame(inventory), "network_plot_inventory")


def main() -> None:
    build_tabular_assets()
    build_network_figure()
    print("SUPPLEMENT ASSETS: PASSED")
    print(f"tables=24")
    print(f"generated_dir={OUT}")
    print(f"network_figure={FIG / 'supplementary_expanded_physical_networks.pdf'}")


if __name__ == "__main__":
    main()
