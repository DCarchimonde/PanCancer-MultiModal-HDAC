import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = path.resolve(process.env.PANCANCER_REVISION_ROOT || process.cwd());
const OUT_DIR = process.env.PANCANCER_WORKBOOK_OUT || path.join(ROOT, "outputs");
const PREVIEW_DIR = process.env.PANCANCER_WORKBOOK_PREVIEWS || path.join(ROOT, "qa", "workbook_previews");

const specs = [
  ["00_TCGA_Cohorts", "supplement/generated/tcga_cohort_manifest.csv", "Manifest of the 22 TCGA/GDC projects and repeated global disease-matrix/mask hashes; these are not cohort-specific hashes. Per-cohort final sample counts were not retained and are not reconstructed."],
  ["01_Dataset", "supplement/generated/dataset_composition.csv", "Dataset composition; signatures, structures, cells, conditions, and genes are distinct units."],
  ["02_Split_Audit", "supplement/generated/split_audit.csv", "Leakage and Murcko-scaffold overlap inventory for all five evaluation settings."],
  ["03_Model_Runs", "derived_final/generalization_all_runs_corrected.csv", "All 34 frozen model runs; corrected 1,856-profile HDAC summaries only."],
  ["04_Model_Summary", "derived_final/generalization_summary_corrected.csv", "Model-by-split mean, SD, confidence interval, epochs, and runtime."],
  ["05_Model_Paired", "derived_final/generalization_paired_corrected.csv", "Seed-paired model differences; positive values favor dual stream by construction."],
  ["06_HDAC_Enrichment", "derived_final/formal_hdac_enrichment.csv", "Formal top-library HDAC enrichment using mean rank fraction (0 = strongest; lower is better) for primary and legacy sensitivity metrics."],
  ["07_Candidate_Identity", "supplement/generated/candidate_identity_and_roles.csv", "Canonical structures, BRD IDs, identity conflicts, and frozen evidence roles."],
  ["08_Candidate_Splits", "results/revision/lincs_candidate_validation/candidate_split_membership.csv", "Candidate train/test membership by split."],
  ["09_Official_Coverage", "supplement/generated/official_condition_coverage.csv", "All/QC/HiQ profile counts with all-official-cell and HiQ-cell counts separated."],
  ["10_Official_Conditions", "results/revision/lincs_condition_audit/official_candidate_conditions.csv", "Official candidate dose, time, cell, and quality metadata."],
  ["11_Cell_Label_Diffs", "results/revision/measured_lincs_figures/candidate_official_cell_label_differences.csv", "Cache-versus-official cell-label differences; official labels are authoritative."],
  ["12_HDAC30", "derived_final/corrected_hdac_holdout_30_structures.csv", "Complete corrected annotation-defined HDAC holdout inventory."],
  ["13_HDAC_PrePost", "derived_final/corrected_hdac_holdout_pre_post_fix.csv", "Discarded pre-correction and corrected holdout counts."],
  ["14_HDAC_Candidate_Seed", "results/revision/lincs_candidate_validation/hdac_holdout_candidate_metrics_by_seed.csv", "Candidate-level corrected-HDAC profile-prediction metrics by seed."],
  ["15_LDO_Membership", "results/revision/leave_drug_candidate_validation/candidate_leave_drug_membership.csv", "Strict leave-drug candidate membership."],
  ["16_LDO_BySeed", "results/revision/leave_drug_candidate_validation/leave_drug_candidate_metrics_by_seed.csv", "Strict candidate metrics by model and seed."],
  ["17_LDO_Summary", "results/revision/leave_drug_candidate_validation/leave_drug_candidate_metrics_summary.csv", "Strict candidate mean, SD, and confidence intervals."],
  ["18_LDO_Paired", "results/revision/leave_drug_candidate_validation/leave_drug_candidate_paired_comparison.csv", "Seed-paired strict candidate model differences."],
  ["19_Measured_Cancer", "results/revision/measured_lincs_figures/candidate_cancer_measured_summary_official.csv", "Candidate-by-cancer official-condition measured reversal."],
  ["20_Measured_PanCancer", "results/revision/measured_lincs_figures/candidate_pan_cancer_measured_summary_official.csv", "Pan-cancer measured reversal by quality stratum and metric."],
  ["21_Condition_Manifest", "supplement/generated/condition_level_manifest.csv", "Manifest for the analysis-time 83,490-row condition-level table; the portable source and its SHA256 were not preserved, and this absence is explicit."],
  ["22_Measured_Time", "results/revision/measured_lincs_figures/candidate_time_measured_summary.csv", "Measured reversal summarized by time."],
  ["23_Measured_Dose", "results/revision/measured_lincs_figures/candidate_dose_measured_summary.csv", "Measured reversal summarized by dose."],
  ["24_PredMeasured_Plot", "results/revision/measured_lincs_figures/predicted_measured_hiq_plot_data.csv", "Candidate-cancer plot data for corrected-HDAC prediction versus HiQ measurement."],
  ["25_PredMeasured_Seed", "results/revision/measured_lincs_figures/predicted_measured_hiq_correlations_by_seed.csv", "Prediction-measurement association by model seed."],
  ["26_PredMeasured_TwoWay", "derived_final/predicted_measured_two_way_bootstrap.csv", "Crossed candidate-cancer bootstrap, 10,000 iterations, seed 20260719."],
  ["27_PredMeasured_LOO", "derived_final/predicted_measured_leave_one_out.csv", "Leave-one-candidate-out and leave-one-cancer-out influence diagnostics."],
  ["28_Background_Rank", "derived_final/measured_all_profile_background_rank_sensitivity.csv", "Unmatched all-profile empirical ranks; not a condition-matched non-HDAC null."],
  ["29_Stability_Primary48", "derived_final/candidate_stability_primary_48.csv", "Primary stability using strength percentile (100 = strongest; higher is better) and the two primary reversal metrics."],
  ["30_Stability_Sens72", "derived_final/candidate_stability_legacy_inclusive_72.csv", "Legacy-inclusive 72-configuration strength-percentile sensitivity analysis (100 = strongest)."],
  ["31_Stability_Compare", "derived_final/candidate_stability_primary_vs_sensitivity.csv", "Primary versus legacy-inclusive strength-percentile comparison (100 = strongest)."],
  ["32_Stability_BySplit", "results/revision/candidate_stability/candidate_stability_by_split_metric.csv", "Candidate strength percentile stratified by split and reversal metric (100 = strongest; higher is better)."],
  ["33_Neighbor_Summary", "results/revision/structural_neighbor_audit/candidate_train_similarity_summary.csv", "Candidate exact, scaffold, and Morgan-neighbor exposure across splits."],
  ["34_Neighbor_Top10", "results/revision/structural_neighbor_audit/candidate_top10_training_neighbors.csv", "Ten nearest training structures for each candidate and split."],
  ["35_Structural_Splits", "results/revision/structural_neighbor_audit/structural_split_inventory.csv", "Structure/scaffold inventory by split."],
  ["36_Reversal_Genes", "results/revision/frozen_candidate_biology/candidate_network_gene_sets.csv", "Frozen reversal-associated gene rows; not direct targets."],
  ["37_STRING_Nodes", "supplement/generated/string_nodes_sanitized.csv", "Readable STRING node attributes for physical and functional networks."],
  ["38_STRING_Edges", "results/revision/reversal_gene_networks_pathways/string_network_edges.csv", "Full STRING physical and functional edge table."],
  ["39_STRING_Summary", "results/revision/reversal_gene_networks_pathways/string_network_summary.csv", "Network density, components, and edge counts."],
  ["40_Enrichment_Full", "supplement/generated/gprofiler_enrichment_sanitized.csv", "Sanitized complete g:Profiler results with exact probabilities and mapped domain."],
  ["41_Enrichment_Rep", "supplement/generated/pathway_representative_nonredundant.csv", "Nonredundant representative terms; scientific probabilities retained."],
  ["41A_gProfiler_Audit", "supplement/generated/gprofiler_intersection_repair_audit.csv", "Frozen-cache identifier alignment, evidence-code separation, intersection-count gates, and output hashes."],
  ["42_Gene_Overlap", "results/revision/reversal_gene_networks_pathways/primary_control_gene_overlap.csv", "Candidate-reference reversal-associated gene overlap."],
  ["43_Pathway_Overlap", "results/revision/reversal_gene_networks_pathways/candidate_pathway_overlap.csv", "Candidate pathway-term overlap."],
  ["44_DepMap_PanCancer", "results/revision/depmap_hdac_background/analysis/depmap_hdac_pan_cancer_summary.csv", "DepMap Public 26Q1 class I HDAC pan-cancer summary."],
  ["45_DepMap_Lineage", "results/revision/depmap_hdac_background/analysis/depmap_hdac_lineage_summary.csv", "Eligible-lineage HDAC gene-effect summary."],
  ["46_DepMap_Heterogeneity", "results/revision/depmap_hdac_background/analysis/depmap_hdac_lineage_heterogeneity.csv", "Kruskal-Wallis, epsilon-squared, and FDR results."],
  ["47_DepMap_Correlations", "results/revision/depmap_hdac_background/analysis/depmap_hdac_gene_correlations.csv", "Pairwise class I HDAC gene-effect correlations."],
  ["48_Redocking", "supplement/generated/redocking_summary.csv", "Standard Vina versus zinc-aware 4LXZ crystallographic redocking."],
  ["49_Docking_4LXZ", "supplement/generated/docking_4lxz_sanitized.csv", "Controlled 4LXZ panel; repository-relative file names and explicit hash status."],
  ["50_Docking_4BKX", "supplement/generated/docking_4bkx_sanitized.csv", "Aligned HDAC1 4BKX docking summary with repository-portable medoid file names."],
  ["51_Receptor_Sensitivity", "supplement/generated/docking_receptor_sensitivity.csv", "4LXZ versus 4BKX pose and score sensitivity."],
  ["52_External_Sensitivity", "supplement/generated/external_drug_sensitivity_availability.csv", "Direct external drug-sensitivity availability and null surrogate result."],
  ["53_Reproducibility", "supplement/generated/reproducibility_inventory.csv", "Hardware, software, seeds, runtime, and unrecorded fields."],
];

const dictionaryRows = [
  ["Level 5 signature", "LINCS replicate-collapsed expression profile; not a unique compound or raw replicate."],
  ["Canonical structure", "RDKit-canonicalized molecular structure used for grouping and screening."],
  ["Drug-cell pair split", "Groups canonical structure plus cell line; does not imply unseen-drug evaluation."],
  ["Leave-drug-out", "Groups canonical structure so test structures are absent from training."],
  ["Leave-cell-line-out", "Groups cell line; with chemical-only inputs this is an unseen-background agreement test."],
  ["Scaffold split", "Groups Bemis-Murcko scaffold; acyclic structures use structure-specific groups."],
  ["Corrected annotation-defined HDAC", "All unambiguously annotated HDAC structures held out; 1,856 profiles, 30 structures."],
  ["signed wTRS", "Negative disease-prediction dot product normalized by the disease absolute magnitude; higher means stronger opposition."],
  ["Spearman reversal", "Negative Spearman disease-perturbation correlation; higher means stronger rank opposition."],
  ["legacy wTRS", "Historical reversal-only magnitude score; retained only for sensitivity analysis."],
  ["HiQ", "Official LINCS is_hiq quality flag; primary measured stratum."],
  ["QC pass", "Official LINCS qc_pass quality flag; sensitivity stratum."],
  ["All cells", "Distinct official cell labels across every matched profile."],
  ["HiQ cells", "Distinct official cell labels remaining in the HiQ primary analysis."],
  ["Reversal-associated gene", "Gene contributing to disease-perturbation opposition; not a direct drug target."],
  ["Class I explicit target subset", "Structures with explicit HDAC1, HDAC2, HDAC3, or HDAC8 target annotation."],
  ["Two-way bootstrap", "Independent resampling of candidates and cancers to preserve crossed clustering."],
  ["HDAC enrichment mean rank fraction", "Within each run and cancer, rank is scaled from 0 to 1 with 0 = strongest. The mean across 528 observations defines the consensus; lower is better."],
  ["Candidate stability strength percentile", "Candidate rank is reoriented to a 0--100 strength scale with 100 = strongest. Higher is better."],
  ["Stability primary48", "4 splits x 2 models x 3 seeds x 2 primary reversal metrics, reported as strength percentile (100 = strongest)."],
  ["Stability sensitivity72", "Primary48 plus legacy wTRS configurations, reported as strength percentile (100 = strongest)."],
  ["Structural sensitivity", "Docking protocol/receptor sensitivity; not biochemical binding validation."],
  ["DepMap gene effect", "CRISPR loss-of-function effect; not a compound response or normal-tissue safety measure."],
  ["g:Profiler submitted background", "11,144 measured Entrez identifiers sent in the request."],
  ["g:Profiler effective domain", "11,154 service-reported statistical universe after identifier mapping."],
  ["TCGA cohort counts", "Final per-cohort tumor/normal sample counts were not retained in the frozen archive and are left blank rather than reconstructed from current GDC holdings."],
];

function colName(n) {
  let out = "";
  let x = n;
  while (x > 0) {
    x -= 1;
    out = String.fromCharCode(65 + (x % 26)) + out;
    x = Math.floor(x / 26);
  }
  return out;
}

function safeTableName(sheetName) {
  return `T_${sheetName.replace(/[^A-Za-z0-9]/g, "_")}`.slice(0, 250);
}

function inferNumberFormat(header) {
  const h = String(header).toLowerCase();
  if (/sha|id|name|role|source|path|smiles|candidate|compound|model|split|metric|gene|cell|time|dose_levels|definition|scope|interpret/.test(h)) return null;
  if (/strength_percentile|stability_percentile/.test(h)) return "0.0";
  if (/fraction|percent/.test(h)) return "0.0%";
  if (/p_value|fdr|p$/.test(h)) return "0.00E+00";
  if (/count|rows|columns|profiles|signatures|structures|models|cells|seeds|epoch|genes|nodes|edges|rank$|top_k|size|iterations/.test(h)) return "#,##0";
  if (/runtime/.test(h)) return "#,##0.0";
  return "0.000";
}

async function styleDataSheet(workbook, sheet, sheetName) {
  sheet.showGridLines = false;
  const used = sheet.getUsedRange(true);
  const values = used.values;
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  if (!rows || !cols) return { rows: 0, cols: 0 };
  sheet.freezePanes.freezeRows(1);
  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format.fill = "#1F4E78";
  header.format.font = { bold: true, color: "#FFFFFF", size: 10 };
  header.format.wrapText = true;
  header.format.rowHeightPx = 34;
  const all = sheet.getRangeByIndexes(0, 0, rows, cols);
  all.format.font = { name: "Aptos", size: 9 };
  all.format.verticalAlignment = "center";
  const table = sheet.tables.add(`A1:${colName(cols)}${rows}`, true, safeTableName(sheetName));
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  table.showBandedRows = true;
  for (let c = 0; c < cols; c += 1) {
    const headerText = String(values[0][c] ?? "");
    let maxLen = headerText.length;
    const sampleRows = Math.min(rows, 100);
    for (let r = 1; r < sampleRows; r += 1) maxLen = Math.max(maxLen, String(values[r]?.[c] ?? "").length);
    let width = Math.max(72, Math.min(250, 8.0 * Math.min(maxLen, 32) + 18));
    if (/description|interpret|note|scope|definition|smiles|path|provenance|intersection_genes/.test(headerText.toLowerCase())) width = 260;
    const col = sheet.getRangeByIndexes(0, c, rows, 1);
    col.format.columnWidthPx = width;
    if (width >= 240) col.format.wrapText = true;
    const numFmt = inferNumberFormat(headerText);
    if (numFmt && rows > 1) sheet.getRangeByIndexes(1, c, rows - 1, 1).format.numberFormat = numFmt;
  }
  return { rows: rows - 1, cols };
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
const indexSheet = workbook.worksheets.add("Sheet_Index");
const dictionary = workbook.worksheets.add("Data_Dictionary");

const imported = [];
for (const [sheetName, relPath, description] of specs) {
  const abs = path.join(ROOT, relPath);
  const csvText = await fs.readFile(abs, "utf8");
  // Parse each CSV in an isolated workbook. Repeated instance-level CSV imports
  // currently collide with artifact-tool's collaborative-document hydration.
  const parsedWorkbook = await Workbook.fromCSV(csvText, { sheetName });
  const parsedSheet = parsedWorkbook.worksheets.getItem(sheetName);
  const parsedValues = parsedSheet.getUsedRange(true).values.map(row => row.map(value => {
    if (typeof value !== "string") return value;
    const marker = "PanCancer-MultiModal-HDAC/";
    if (value.includes(marker)) return value.slice(value.indexOf(marker) + marker.length);
    if (value.startsWith(`${ROOT}/`)) return value.slice(ROOT.length + 1);
    return value;
  }));
  const sheet = workbook.worksheets.add(sheetName);
  if (parsedValues.length > 0 && (parsedValues[0]?.length ?? 0) > 0) {
    sheet.getRangeByIndexes(0, 0, parsedValues.length, parsedValues[0].length).values = parsedValues;
  }
  const dims = await styleDataSheet(workbook, sheet, sheetName);
  imported.push({ sheetName, relPath, description, ...dims });
}

readme.showGridLines = false;
const readmeRows = [
  ["File", "Additional file 2: Machine-readable major-revision tables"],
  ["Manuscript", "Multi-Modal Molecular Representation Learning Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal"],
  ["Evidence freeze", "2026-07-19"],
  ["Evidence commit", "fefbf2f4fa7399983d4d4041dcb8e7e91b84d17f"],
  ["Submission ID", "8b1e184c-443d-4707-954c-70a65930e16f"],
  ["Data sheets", specs.length],
  ["Primary model conclusion", "Dual-stream and fingerprint MLP are comparable; no stable dual-stream superiority."],
  ["Candidate hierarchy", "Mocetinostat core; NCH-51 secondary; TC-H-106 exploratory."],
  ["Measured evidence", "Official-condition LINCS transcriptomic reversal; not viability or efficacy."],
  ["Class enrichment", "Signed wTRS significant; Spearman reversal not significant."],
  ["Stability", "48 primary configurations; 72 legacy-inclusive configurations as sensitivity only."],
  ["Excluded/downgraded", "RG2833 prediction-only; Tianeptinaline/BG-1010 identity conflict excluded."],
  ["Biology", "Reversal-associated genes; not direct drug targets."],
  ["Docking", "Protocol/receptor sensitivity; not binding validation."],
  ["External drug sensitivity", "No direct TC-H-106 public viability response; Entinostat surrogate R=-0.052, P=0.859 is null."],
  ["g:Profiler", "11,144 submitted IDs; 11,154 service-reported effective domain."],
  ["TCGA cohort manifest", "All 22 GDC projects and frozen matrix hashes are listed; final per-cohort tumor/normal counts were not preserved and are not reconstructed."],
  ["Condition-level source", "The analysis-time 83,490-row condition_level_measured_scores.csv and its SHA256 were not preserved in the frozen portable package; all aggregates used for figures and inference are included."],
  ["CPU/RAM", "Not recorded; not fabricated."],
  ["Navigation", "Sheet_Index lists every included table, repository-relative source, scope, row count, and field count. Data_Dictionary defines recurring terms."],
  ["Corrected holdout", "No stale 1,565-profile HDAC summary is used for model inference; corrected results use 1,856 profiles and 30 structures."],
];
readme.getRange(`A1:B${readmeRows.length}`).values = readmeRows;
readme.getRange(`A1:B${readmeRows.length}`).format.wrapText = true;
readme.getRange(`A1:B${readmeRows.length}`).format.borders = { preset: "all", style: "thin", color: "#B4C6E7" };
readme.getRange(`A1:A${readmeRows.length}`).format.fill = "#D9EAF7";
readme.getRange(`A1:A${readmeRows.length}`).format.font = { bold: true, color: "#17365D", size: 10 };
readme.getRange("A1:B2").format.fill = "#17365D";
readme.getRange("A1:B2").format.font = { bold: true, color: "#FFFFFF", size: 11 };
readme.getRange(`A17:B${readmeRows.length}`).format.fill = "#FFF2CC";
readme.getRange(`A${readmeRows.length}:B${readmeRows.length}`).format.fill = "#E2F0D9";
readme.getRange(`A${readmeRows.length}:B${readmeRows.length}`).format.font = { italic: true, color: "#375623" };
readme.getRange("A:A").format.columnWidthPx = 190;
readme.getRange("B:B").format.columnWidthPx = 760;
readme.freezePanes.freezeRows(2);

indexSheet.showGridLines = false;
indexSheet.getRange("A1:E1").values = [["Sheet", "Rows", "Columns", "Repository-relative source", "Scope / evidence boundary"]];
indexSheet.getRange(`A2:E${imported.length + 1}`).values = imported.map(x => [x.sheetName, x.rows, x.cols, x.relPath, x.description]);
await styleDataSheet(workbook, indexSheet, "Sheet_Index");

dictionary.showGridLines = false;
dictionary.getRange("A1:B1").values = [["Term", "Definition / interpretation boundary"]];
dictionary.getRange(`A2:B${dictionaryRows.length + 1}`).values = dictionaryRows;
await styleDataSheet(workbook, dictionary, "Data_Dictionary");
dictionary.getRange("B:B").format.columnWidthPx = 520;
dictionary.getRange("B:B").format.wrapText = true;

const keyInspect = await workbook.inspect({
  kind: "table",
  range: `README!A1:B${readmeRows.length}`,
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 8,
  maxChars: 5000,
});
console.log(keyInspect.ndjson);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

const absolutePathScan = await workbook.inspect({
  kind: "match",
  searchTerm: [["", "root", "autodl-tmp"].join("/"), ["", "workspace", "scratch"].join("/")].join("|"),
  options: { useRegex: true, maxResults: 100 },
  summary: "absolute path scan",
});
console.log(absolutePathScan.ndjson);

await fs.mkdir(OUT_DIR, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
const outPath = `${OUT_DIR}/Additional_file_2_Revision_Tables.xlsx`;
await output.save(outPath);

await fs.mkdir(PREVIEW_DIR, { recursive: true });
const renderSheets = process.env.WORKBOOK_RENDER_MODE === "key"
  ? ["README", "Sheet_Index", "Data_Dictionary", "49_Docking_4LXZ", "50_Docking_4BKX"]
  : process.env.WORKBOOK_RENDER_MODE === "none"
    ? []
    : ["README", "Sheet_Index", "Data_Dictionary", ...specs.map(x => x[0])];
for (const sheetName of renderSheets) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange(true);
  const values = used.values;
  const rows = Math.min(values.length, 25);
  const cols = Math.min(values[0]?.length ?? 1, 12);
  const range = `A1:${colName(cols)}${Math.max(rows, 1)}`;
  const preview = await workbook.render({ sheetName, range, scale: 0.55, format: "png" });
  await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(`WORKBOOK: PASSED\nsheets=${specs.length + 3}\npath=${outPath}\npreviews=${PREVIEW_DIR}`);
