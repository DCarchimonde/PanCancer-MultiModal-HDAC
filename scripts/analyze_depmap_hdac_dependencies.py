#!/usr/bin/env python3
"""Analyze frozen DepMap 26Q1 HDAC genetic-dependency context.

The analysis is deliberately descriptive and lineage-aware.  It does not use
CRISPR gene effect as drug-response validation, and it does not infer a normal-
tissue therapeutic window because DepMap contains cancer models only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import TwoSlopeNorm
from scipy import stats


GENES = ("HDAC1", "HDAC2", "HDAC3", "HDAC8")
DEFAULT_SOURCE_DIR = Path("results/revision/depmap_hdac_background/source_subset")
DEFAULT_OUTPUT_DIR = Path("results/revision/depmap_hdac_background/analysis")
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
DEFAULT_RANDOM_SEED = 20260718
DEFAULT_MIN_PRIMARY_LINEAGE_MODELS = 10
DESCRIPTIVE_THRESHOLDS = (-0.5, -1.0)
COLORS = {
    "HDAC1": "#176B87",
    "HDAC2": "#2E86AB",
    "HDAC3": "#C0392B",
    "HDAC8": "#D68910",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--min-primary-lineage-models",
        type=int,
        default=DEFAULT_MIN_PRIMARY_LINEAGE_MODELS,
        help="Minimum models for primary lineage heatmap and heterogeneity test.",
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unique_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} matching {directory / pattern}; "
            f"observed={[str(path) for path in matches]}"
        )
    return matches[0]


def stable_seed(base_seed: int, *values: str) -> int:
    payload = "\x1f".join(values).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return int((base_seed + offset) % (2**32 - 1))


def bootstrap_median_ci(
    values: np.ndarray, iterations: int, seed: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    medians = np.empty(iterations, dtype=float)
    chunk_size = max(1, min(iterations, 250))
    position = 0
    while position < iterations:
        current = min(chunk_size, iterations - position)
        indices = rng.integers(0, values.size, size=(current, values.size))
        medians[position : position + current] = np.median(values[indices], axis=1)
        position += current
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.dropna().astype(float)
    if finite.empty:
        return result
    ordered = finite.sort_values()
    count = len(ordered)
    adjusted = ordered.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result.loc[ordered.index] = adjusted
    return result


def validate_and_join(
    gene_path: Path, model_path: Path
) -> tuple[pd.DataFrame, str]:
    gene_effect = pd.read_csv(gene_path, dtype={"ModelID": "string"})
    model = pd.read_csv(model_path, dtype={"ModelID": "string"}, low_memory=False)
    required_gene = {"ModelID", *GENES}
    missing_gene = required_gene - set(gene_effect.columns)
    if missing_gene:
        raise RuntimeError(f"Frozen gene-effect subset is missing {sorted(missing_gene)}")
    if "ModelID" not in model.columns:
        raise RuntimeError("Frozen model metadata is missing ModelID")
    for label, frame in (("gene effect", gene_effect), ("model metadata", model)):
        frame["ModelID"] = frame["ModelID"].astype("string").str.strip()
        if frame["ModelID"].isna().any() or (frame["ModelID"] == "").any():
            raise RuntimeError(f"{label} contains missing ModelID values")
        if frame["ModelID"].duplicated().any():
            raise RuntimeError(f"{label} contains duplicate ModelID values")
    if not gene_effect["ModelID"].str.fullmatch(r"ACH-\d{6}", na=False).all():
        raise RuntimeError("Frozen gene-effect subset contains malformed ModelID values")
    for gene in GENES:
        gene_effect[gene] = pd.to_numeric(gene_effect[gene], errors="coerce")
        if not np.isfinite(gene_effect[gene].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{gene} contains non-finite values in the frozen subset")
    merged = gene_effect.merge(
        model,
        on="ModelID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise RuntimeError(f"Model metadata join left {unmatched} unmatched gene-effect rows")
    merged = merged.drop(columns="_merge")
    lineage_candidates = (
        "OncotreeLineage",
        "OncotreePrimaryDisease",
        "DepmapModelType",
    )
    lineage_column = next(
        (
            column
            for column in lineage_candidates
            if column in merged.columns
            and merged[column].astype("string").str.strip().replace("", pd.NA).notna().any()
        ),
        None,
    )
    if lineage_column is None:
        raise RuntimeError(
            f"No usable lineage field found; checked={list(lineage_candidates)}"
        )
    lineage = merged[lineage_column].astype("string").str.strip()
    lineage = lineage.mask(lineage.isna() | (lineage == ""), "Unannotated")
    merged.insert(1, "AnalysisLineage", lineage)
    return merged, lineage_column


def descriptive_row(
    values: np.ndarray,
    *,
    gene: str,
    lineage: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    low, high = bootstrap_median_ci(values, iterations, seed)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {
        "lineage": lineage,
        "gene": gene,
        "models": int(values.size),
        "mean_gene_effect": float(np.mean(values)),
        "sd_gene_effect": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
        "median_gene_effect": float(np.median(values)),
        "median_ci95_low": low,
        "median_ci95_high": high,
        "q25_gene_effect": float(q25),
        "q75_gene_effect": float(q75),
        "iqr_gene_effect": float(q75 - q25),
        "fraction_gene_effect_le_neg_0_5": float(np.mean(values <= -0.5)),
        "fraction_gene_effect_le_neg_1_0": float(np.mean(values <= -1.0)),
    }


def summarize_dependencies(
    merged: pd.DataFrame, iterations: int, base_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pan_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for gene in GENES:
        pan_rows.append(
            descriptive_row(
                merged[gene].to_numpy(dtype=float),
                gene=gene,
                lineage="Pan-cancer",
                iterations=iterations,
                seed=stable_seed(base_seed, "pan-cancer", gene),
            )
        )
    for lineage, group in merged.groupby("AnalysisLineage", sort=True, dropna=False):
        for gene in GENES:
            lineage_rows.append(
                descriptive_row(
                    group[gene].to_numpy(dtype=float),
                    gene=gene,
                    lineage=str(lineage),
                    iterations=iterations,
                    seed=stable_seed(base_seed, str(lineage), gene),
                )
            )
    return pd.DataFrame(pan_rows), pd.DataFrame(lineage_rows)


def lineage_heterogeneity(
    merged: pd.DataFrame, min_models: int
) -> pd.DataFrame:
    counts = merged.groupby("AnalysisLineage", sort=True).size()
    eligible = counts[counts >= min_models].index.tolist()
    rows: list[dict[str, Any]] = []
    for gene in GENES:
        groups = [
            merged.loc[merged["AnalysisLineage"] == lineage, gene].to_numpy(dtype=float)
            for lineage in eligible
        ]
        groups = [values[np.isfinite(values)] for values in groups if values.size]
        if len(groups) < 2:
            statistic = p_value = effect = math.nan
            total = sum(len(values) for values in groups)
        else:
            result = stats.kruskal(*groups, nan_policy="omit")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            total = sum(len(values) for values in groups)
            denominator = total - len(groups)
            effect = (
                float(max(0.0, (statistic - len(groups) + 1) / denominator))
                if denominator > 0
                else math.nan
            )
        rows.append(
            {
                "gene": gene,
                "eligible_lineages": len(groups),
                "models": total,
                "kruskal_wallis_h": statistic,
                "p_value": p_value,
                "epsilon_squared": effect,
            }
        )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = benjamini_hochberg(frame["p_value"])
    return frame


def gene_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for first, second in combinations(GENES, 2):
        values = merged[[first, second]].dropna()
        result = stats.spearmanr(values[first], values[second])
        rows.append(
            {
                "gene_a": first,
                "gene_b": second,
                "models": int(len(values)),
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = benjamini_hochberg(frame["p_value"])
    return frame


def plot_overall_distributions(
    merged: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(7.8, 5.0))
    values = [merged[gene].to_numpy(dtype=float) for gene in GENES]
    violin = axis.violinplot(
        values,
        positions=np.arange(len(GENES)),
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=300,
        bw_method="scott",
    )
    for body, gene in zip(violin["bodies"], GENES):
        body.set_facecolor(COLORS[gene])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.72)
    box = axis.boxplot(
        values,
        positions=np.arange(len(GENES)),
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    for patch, gene in zip(box["boxes"], GENES):
        patch.set_facecolor(COLORS[gene])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
    axis.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    axis.axhline(-0.5, color="#777777", linestyle="--", linewidth=0.9)
    axis.axhline(-1.0, color="#444444", linestyle=":", linewidth=1.0)
    axis.set_xticks(np.arange(len(GENES)), GENES)
    axis.set_ylabel("DepMap 26Q1 CRISPR gene-effect score")
    axis.set_title("Pan-cancer HDAC genetic-dependency distributions")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output_dir / "depmap_hdac_overall_distributions.png", dpi=dpi)
    figure.savefig(output_dir / "depmap_hdac_overall_distributions.pdf")
    plt.close(figure)


def plot_lineage_heatmap(
    lineage_summary: pd.DataFrame,
    output_dir: Path,
    min_models: int,
    dpi: int,
) -> list[str]:
    counts = (
        lineage_summary.groupby("lineage", sort=True)["models"].max().astype(int)
    )
    eligible = counts[counts >= min_models].index.tolist()
    if len(eligible) < 2:
        raise RuntimeError(
            f"Fewer than two lineages have at least {min_models} models"
        )
    selected = lineage_summary.loc[lineage_summary["lineage"].isin(eligible)]
    matrix = selected.pivot(index="lineage", columns="gene", values="median_gene_effect")
    matrix = matrix.loc[:, list(GENES)]
    ordering = matrix.mean(axis=1).sort_values().index
    matrix = matrix.loc[ordering]
    labels = [f"{lineage} (n={counts[lineage]})" for lineage in matrix.index]
    vmin = min(-1.0, float(np.nanmin(matrix.to_numpy())))
    vmax = max(0.2, float(np.nanmax(matrix.to_numpy())))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    height = max(5.0, 0.34 * len(matrix) + 1.7)
    figure, axis = plt.subplots(figsize=(7.3, height))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
    axis.set_xticks(np.arange(len(GENES)), GENES)
    axis.set_yticks(np.arange(len(matrix)), labels)
    axis.set_title(
        f"Median HDAC gene effect by OncoTree lineage (n ≥ {min_models})"
    )
    for row in range(len(matrix)):
        for column in range(len(GENES)):
            value = float(matrix.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(norm(value) - 0.5) > 0.28 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.75, pad=0.03)
    colorbar.set_label("Median CRISPR gene-effect score")
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "depmap_hdac_lineage_heatmap.png", dpi=dpi)
    figure.savefig(output_dir / "depmap_hdac_lineage_heatmap.pdf")
    plt.close(figure)
    return list(matrix.index)


def plot_correlation_heatmap(
    merged: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    matrix = merged.loc[:, list(GENES)].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(5.4, 4.7))
    image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(GENES)), GENES)
    axis.set_yticks(np.arange(len(GENES)), GENES)
    for row in range(len(GENES)):
        for column in range(len(GENES)):
            value = float(matrix.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.78, pad=0.04)
    colorbar.set_label("Spearman correlation")
    axis.set_title("HDAC CRISPR co-dependency across cancer models")
    figure.tight_layout()
    figure.savefig(output_dir / "depmap_hdac_gene_correlation.png", dpi=dpi)
    figure.savefig(output_dir / "depmap_hdac_gene_correlation.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.bootstrap_iterations < 1000:
        raise ValueError("--bootstrap-iterations must be at least 1000")
    if args.min_primary_lineage_models < 5:
        raise ValueError("--min-primary-lineage-models must be at least 5")
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gene_path = require_unique_file(
        args.source_dir,
        "*_CRISPRGeneEffect_HDAC1_2_3_8.csv",
        "frozen HDAC gene-effect subset",
    )
    model_path = require_unique_file(
        args.source_dir, "*_Model_metadata.csv", "frozen model metadata subset"
    )
    source_manifest_path = require_unique_file(
        args.source_dir,
        "depmap_26q1_hdac_subset_manifest.json",
        "source-subset manifest",
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "passed":
        raise RuntimeError("DepMap source subset manifest is not passed")
    if source_manifest.get("release") != "DepMap Public 26Q1":
        raise RuntimeError(
            f"Unexpected frozen DepMap release: {source_manifest.get('release')!r}"
        )
    if tuple(source_manifest.get("genes", [])) != GENES:
        raise RuntimeError(
            f"Unexpected frozen genes: {source_manifest.get('genes')!r}"
        )

    merged, lineage_column = validate_and_join(gene_path, model_path)
    pan_summary, lineage_summary = summarize_dependencies(
        merged, args.bootstrap_iterations, args.random_seed
    )
    heterogeneity = lineage_heterogeneity(
        merged, args.min_primary_lineage_models
    )
    correlations = gene_correlations(merged)

    model_columns = [
        "ModelID",
        "AnalysisLineage",
        *GENES,
        *[
            column
            for column in (
                "CellLineName",
                "DepmapModelType",
                "OncotreeLineage",
                "OncotreePrimaryDisease",
                "OncotreeSubtype",
                "OncotreeCode",
                "Sex",
                "PrimaryOrMetastasis",
                "ModelType",
            )
            if column in merged.columns
        ],
    ]
    model_level = merged.loc[:, list(dict.fromkeys(model_columns))].copy()
    model_level = model_level.sort_values("ModelID", kind="stable").reset_index(drop=True)
    pan_summary.to_csv(args.output_dir / "depmap_hdac_pan_cancer_summary.csv", index=False)
    lineage_summary.to_csv(args.output_dir / "depmap_hdac_lineage_summary.csv", index=False)
    heterogeneity.to_csv(
        args.output_dir / "depmap_hdac_lineage_heterogeneity.csv", index=False
    )
    correlations.to_csv(args.output_dir / "depmap_hdac_gene_correlations.csv", index=False)
    model_level.to_csv(args.output_dir / "depmap_hdac_model_level.csv", index=False)

    plot_overall_distributions(merged, args.output_dir, args.figure_dpi)
    primary_lineages = plot_lineage_heatmap(
        lineage_summary,
        args.output_dir,
        args.min_primary_lineage_models,
        args.figure_dpi,
    )
    plot_correlation_heatmap(merged, args.output_dir, args.figure_dpi)

    output_names = [
        "depmap_hdac_pan_cancer_summary.csv",
        "depmap_hdac_lineage_summary.csv",
        "depmap_hdac_lineage_heterogeneity.csv",
        "depmap_hdac_gene_correlations.csv",
        "depmap_hdac_model_level.csv",
        "depmap_hdac_overall_distributions.png",
        "depmap_hdac_overall_distributions.pdf",
        "depmap_hdac_lineage_heatmap.png",
        "depmap_hdac_lineage_heatmap.pdf",
        "depmap_hdac_gene_correlation.png",
        "depmap_hdac_gene_correlation.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release": "DepMap Public 26Q1",
        "genes": list(GENES),
        "models": int(len(merged)),
        "lineage_field": lineage_column,
        "annotated_lineages": int(merged["AnalysisLineage"].nunique()),
        "primary_lineage_min_models": args.min_primary_lineage_models,
        "primary_lineages": primary_lineages,
        "bootstrap": {
            "unit": "cell-line model",
            "iterations": args.bootstrap_iterations,
            "random_seed": args.random_seed,
            "interval": "percentile 95% CI for the median",
        },
        "descriptive_score_thresholds": list(DESCRIPTIVE_THRESHOLDS),
        "threshold_policy": (
            "Fractions at -0.5 and -1.0 are descriptive summaries of normalized "
            "gene-effect scores, not clinical or drug-response cutoffs."
        ),
        "lineage_test": (
            "Kruskal-Wallis across native OncoTree lineages meeting the pre-specified "
            "minimum model count; Benjamini-Hochberg correction across four HDAC genes."
        ),
        "source_sha256": {
            gene_path.name: sha256_file(gene_path),
            model_path.name: sha256_file(model_path),
            source_manifest_path.name: sha256_file(source_manifest_path),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
        "gpu_required": False,
        "interpretation_guardrail": (
            "DepMap CRISPR gene effect provides genetic-dependency context only. It is "
            "not drug-response validation, does not prove that any candidate binds or "
            "acts through an HDAC, and cannot establish a normal-tissue therapeutic "
            "window because this release panel contains cancer models."
        ),
        "outputs": [
            {
                "path": name,
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in output_names
        ],
    }
    manifest_path = args.output_dir / "depmap_hdac_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("===== DEPMAP PAN-CANCER HDAC DEPENDENCY =====")
    print(
        pan_summary[
            [
                "gene",
                "models",
                "median_gene_effect",
                "median_ci95_low",
                "median_ci95_high",
                "fraction_gene_effect_le_neg_0_5",
                "fraction_gene_effect_le_neg_1_0",
            ]
        ].to_string(index=False)
    )
    print("\n===== LINEAGE HETEROGENEITY =====")
    print(heterogeneity.to_string(index=False))
    print("\n===== HDAC GENE-EFFECT CORRELATIONS =====")
    print(correlations.to_string(index=False))
    print("\n===== PRIMARY LINEAGES =====")
    print(f"lineage_field={lineage_column}")
    print(f"minimum_models={args.min_primary_lineage_models}")
    print(" | ".join(primary_lineages))
    final_line = "DEPMAP HDAC DEPENDENCY ANALYSIS: PASSED"
    print("\n" + final_line)

    audit = (
        pan_summary.to_string(index=False)
        + "\n\n"
        + heterogeneity.to_string(index=False)
        + "\n\n"
        + correlations.to_string(index=False)
        + "\n\n"
        + final_line
        + "\n"
        + f"manifest_sha256={sha256_file(manifest_path)}\n"
    )
    (args.output_dir / "audit.log").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
