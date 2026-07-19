#!/usr/bin/env python3
"""Build the final, leakage-aware revision statistics from frozen outputs.

This script performs CPU-only post-processing.  It never retrains a model and
does not alter the frozen candidate roles.  It replaces stale cross-file
summaries with values read directly from the corrected run manifests.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, spearmanr, t
from scipy.stats.contingency import odds_ratio


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision"
OUTPUT = ROOT / "derived_final"
FIGURES = ROOT / "figures"
SEED = 20260719
BOOTSTRAPS = 10_000

PRIMARY_METRICS = ("signed_wtrs", "spearman_reversal")
ALL_STABILITY_METRICS = ("legacy_wtrs",) + PRIMARY_METRICS
SPLIT_LABELS = {
    "pair_split": "Drug--cell pair",
    "leave_drug_out": "Leave drug out",
    "leave_cell_line_out": "Leave cell line out",
    "scaffold_split": "Scaffold split",
    "hdac_class_holdout": "Corrected annotation-defined HDAC holdout",
}
MODEL_LABELS = {
    "dual_stream": "Dual stream",
    "fingerprint_mlp": "Fingerprint MLP",
}
CANDIDATE_ORDER = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
    "RG2833",
    "Tianeptinaline_or_BG-1010",
]
PRIMARY_DISPLAY = CANDIDATE_ORDER[:8]
ROLE_COLORS = {
    "core_candidate": "#9c2f2f",
    "secondary_candidate": "#d97835",
    "original_method_sensitive": "#d9ad3f",
    "class_support": "#4f7ea8",
    "reference_control": "#6f6f6f",
    "prediction_only": "#9f8fbd",
    "identity_conflict_excluded": "#b5b5b5",
}


def bh_fdr(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def corrected_generalization() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    benchmark_root = RESULTS / "benchmarks"
    for split in SPLIT_LABELS:
        for model in MODEL_LABELS:
            for path in sorted((benchmark_root / split / model).glob("seed_*/summary.json")):
                with path.open() as handle:
                    row = json.load(handle)
                row["summary_path"] = str(path.relative_to(ROOT))
                rows.append(row)
    runs = pd.DataFrame(rows).sort_values(["split", "model", "seed"])
    runs.to_csv(OUTPUT / "generalization_all_runs_corrected.csv", index=False)

    metrics = ["mse", "mae", "r2", "pearson_mean", "spearman_mean"]
    summaries: list[dict[str, object]] = []
    for (split, model), subset in runs.groupby(["split", "model"], sort=False):
        record: dict[str, object] = {
            "split": split,
            "split_label": SPLIT_LABELS[split],
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seeds": int(subset["seed"].nunique()),
            "test_profiles": int(subset["test_profiles"].iloc[0]),
        }
        n = len(subset)
        critical = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
        for metric in metrics + ["best_epoch", "runtime_seconds"]:
            values = subset[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            half = critical * sd / math.sqrt(n) if n > 1 else np.nan
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_ci95_low"] = mean - half
            record[f"{metric}_ci95_high"] = mean + half
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "generalization_summary_corrected.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    for split in SPLIT_LABELS:
        wide = runs.loc[runs["split"] == split].set_index(["seed", "model"])
        seeds = sorted(runs.loc[runs["split"] == split, "seed"].unique())
        for metric in metrics:
            diffs = []
            for seed in seeds:
                dual = float(wide.loc[(seed, "dual_stream"), metric])
                fp = float(wide.loc[(seed, "fingerprint_mlp"), metric])
                diffs.append(fp - dual if metric in {"mse", "mae"} else dual - fp)
            values = np.asarray(diffs, dtype=float)
            n = len(values)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
            paired_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "seeds": n,
                    "advantage_dual_mean": mean,
                    "advantage_dual_sd": sd,
                    "advantage_dual_ci95_low": mean - half,
                    "advantage_dual_ci95_high": mean + half,
                    "orientation": "positive values favor dual_stream",
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT / "generalization_paired_corrected.csv", index=False)
    return runs, summary, paired


def two_way_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RESULTS / "measured_lincs_figures" / "predicted_measured_hiq_plot_data.csv"
    data = pd.read_csv(path)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        for model in MODEL_LABELS:
            subset = data.loc[(data["metric"] == metric) & (data["model"] == model)]
            candidates = sorted(subset["candidate"].unique())
            cancers = sorted(subset["cancer"].unique())
            predicted = subset.pivot(
                index="candidate", columns="cancer", values="predicted_score_mean"
            ).loc[candidates, cancers].to_numpy(dtype=float)
            measured = subset.pivot(
                index="candidate", columns="cancer", values="measured_cell_score_median"
            ).loc[candidates, cancers].to_numpy(dtype=float)
            if predicted.shape != (8, 22) or not np.isfinite(predicted).all() or not np.isfinite(measured).all():
                raise RuntimeError(f"Incomplete 8 x 22 matrix for {metric}/{model}")
            point_pearson, point_spearman = safe_correlations(predicted.ravel(), measured.ravel())
            boot = np.empty((BOOTSTRAPS, 2), dtype=float)
            for index in range(BOOTSTRAPS):
                candidate_index = rng.integers(0, len(candidates), len(candidates))
                cancer_index = rng.integers(0, len(cancers), len(cancers))
                pred_sample = predicted[np.ix_(candidate_index, cancer_index)].ravel()
                meas_sample = measured[np.ix_(candidate_index, cancer_index)].ravel()
                boot[index] = safe_correlations(pred_sample, meas_sample)
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "candidates": len(candidates),
                    "cancers": len(cancers),
                    "candidate_cancer_pairs": predicted.size,
                    "pearson": point_pearson,
                    "pearson_two_way_ci95_low": float(np.nanquantile(boot[:, 0], 0.025)),
                    "pearson_two_way_ci95_high": float(np.nanquantile(boot[:, 0], 0.975)),
                    "spearman": point_spearman,
                    "spearman_two_way_ci95_low": float(np.nanquantile(boot[:, 1], 0.025)),
                    "spearman_two_way_ci95_high": float(np.nanquantile(boot[:, 1], 0.975)),
                    "bootstrap_iterations": BOOTSTRAPS,
                    "bootstrap_seed": SEED,
                    "bootstrap_unit": "candidate_and_cancer_crossed_resampling",
                }
            )
            for index, candidate in enumerate(candidates):
                p, s = safe_correlations(
                    np.delete(predicted, index, axis=0).ravel(),
                    np.delete(measured, index, axis=0).ravel(),
                )
                loo_rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "omission_type": "candidate",
                        "omitted": candidate,
                        "pearson": p,
                        "spearman": s,
                    }
                )
            for index, cancer in enumerate(cancers):
                p, s = safe_correlations(
                    np.delete(predicted, index, axis=1).ravel(),
                    np.delete(measured, index, axis=1).ravel(),
                )
                loo_rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "omission_type": "cancer",
                        "omitted": cancer,
                        "pearson": p,
                        "spearman": s,
                    }
                )
    correlations = pd.DataFrame(rows)
    loo = pd.DataFrame(loo_rows)
    correlations.to_csv(OUTPUT / "predicted_measured_two_way_bootstrap.csv", index=False)
    loo.to_csv(OUTPUT / "predicted_measured_leave_one_out.csv", index=False)
    return correlations, loo


def primary_stability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(RESULTS / "candidate_stability" / "candidate_run_stability.csv")

    def summarize(metrics: tuple[str, ...], label: str) -> pd.DataFrame:
        subset = source.loc[source["metric"].isin(metrics)].copy()
        rows: list[dict[str, object]] = []
        for (candidate, role), group in subset.groupby(["candidate", "candidate_role"]):
            values = group["median_library_percentile"].to_numpy(dtype=float)
            grouped = group.groupby(["split", "metric"])["median_library_percentile"].median()
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_role": role,
                    "scope": label,
                    "splits": int(group["split"].nunique()),
                    "metrics": " | ".join(metrics),
                    "models": int(group["model"].nunique()),
                    "seeds": int(group["seed"].nunique()),
                    "run_configurations": len(group),
                    "median_run_library_percentile": float(np.median(values)),
                    "mean_run_library_percentile": float(np.mean(values)),
                    "run_library_percentile_q25": float(np.quantile(values, 0.25)),
                    "run_library_percentile_q75": float(np.quantile(values, 0.75)),
                    "run_library_percentile_sd": float(np.std(values, ddof=1)),
                    "worst_split_metric_median_percentile": float(grouped.min()),
                    "best_split_metric_median_percentile": float(grouped.max()),
                    "top100_observation_fraction": float(group["top100_cancer_fraction"].mean()),
                    "top500_observation_fraction": float(group["top500_cancer_fraction"].mean()),
                    "top1_percent_observation_fraction": float(group["top1_percent_cancer_fraction"].mean()),
                    "top10_percent_observation_fraction": float(group["top10_percent_cancer_fraction"].mean()),
                }
            )
        return pd.DataFrame(rows)

    primary = summarize(PRIMARY_METRICS, "primary_48_config")
    sensitivity = summarize(ALL_STABILITY_METRICS, "legacy_inclusive_72_config_sensitivity")
    primary["rank_within_all_rows"] = primary["median_run_library_percentile"].rank(
        ascending=False, method="min"
    ).astype(int)
    sensitivity["rank_within_all_rows"] = sensitivity["median_run_library_percentile"].rank(
        ascending=False, method="min"
    ).astype(int)
    comparison = primary.merge(
        sensitivity,
        on=["candidate", "candidate_role"],
        suffixes=("_primary48", "_sensitivity72"),
    )
    comparison["median_percentile_change_primary_minus_sensitivity"] = (
        comparison["median_run_library_percentile_primary48"]
        - comparison["median_run_library_percentile_sensitivity72"]
    )
    primary.to_csv(OUTPUT / "candidate_stability_primary_48.csv", index=False)
    sensitivity.to_csv(OUTPUT / "candidate_stability_legacy_inclusive_72.csv", index=False)
    comparison.to_csv(OUTPUT / "candidate_stability_primary_vs_sensitivity.csv", index=False)
    return source, primary, comparison


def formal_hdac_enrichment() -> pd.DataFrame:
    files = {
        "legacy_wtrs": RESULTS / "screening_analysis" / "compound_consensus_legacy_wtrs.csv",
        "signed_wtrs": RESULTS / "screening_analysis" / "compound_consensus_signed_wtrs.csv",
        "spearman_reversal": RESULTS / "spearman_analysis" / "compound_consensus_spearman_reversal.csv",
    }
    thresholds = (0.005, 0.01, 0.05, 0.10)
    rows: list[dict[str, object]] = []
    for metric, path in files.items():
        data = pd.read_csv(path, low_memory=False).sort_values("consensus_rank").reset_index(drop=True)
        hdac = data["is_hdac_annotated"].fillna(False).astype(bool)
        class_i = data["target"].fillna("").str.contains(
            r"(?:^|\|\s*)HDAC(?:1|2|3|8)(?:\s*\||$)", regex=True
        )
        for definition, mask in (("annotation_defined_HDAC", hdac), ("explicit_class_I_target", class_i)):
            total_positive = int(mask.sum())
            for fraction in thresholds:
                top_k = int(math.ceil(len(data) * fraction))
                observed = int(mask.iloc[:top_k].sum())
                table = [
                    [observed, top_k - observed],
                    [total_positive - observed, len(data) - top_k - total_positive + observed],
                ]
                fisher_or, fisher_p = fisher_exact(table, alternative="greater")
                conditional = odds_ratio(table, kind="conditional")
                ci = conditional.confidence_interval(confidence_level=0.95)
                expected = top_k * total_positive / len(data)
                rows.append(
                    {
                        "metric": metric,
                        "metric_role": "primary" if metric in PRIMARY_METRICS else "sensitivity",
                        "annotation_definition": definition,
                        "top_fraction": fraction,
                        "top_k": top_k,
                        "library_compounds": len(data),
                        "library_annotated": total_positive,
                        "observed_annotated": observed,
                        "expected_annotated": expected,
                        "fold_enrichment": observed / expected if expected else np.nan,
                        "fisher_odds_ratio": fisher_or,
                        "conditional_odds_ratio": conditional.statistic,
                        "conditional_or_ci95_low": ci.low,
                        "conditional_or_ci95_high": ci.high,
                        "fisher_one_sided_p": fisher_p,
                    }
                )
    result = pd.DataFrame(rows)
    result["fdr_bh_all_24_tests"] = bh_fdr(result["fisher_one_sided_p"])
    result["fdr_bh_within_metric_definition"] = result.groupby(
        ["metric", "annotation_definition"], group_keys=False
    )["fisher_one_sided_p"].transform(lambda x: bh_fdr(x))
    result["ranking_source"] = "cross_generalization_mean_rank_percentile"
    result["included_splits"] = (
        "leave_drug_out | leave_cell_line_out | scaffold_split | hdac_class_holdout"
    )
    result["included_models"] = "dual_stream | fingerprint_mlp"
    result["included_seeds"] = "1 | 2 | 3"
    result["screening_runs"] = 24
    result["cancers"] = 22
    result["run_cancer_observations_per_compound"] = 528
    result["aggregation_definition"] = (
        "Rank compounds within each run and cancer; convert to library percentile; "
        "average all 528 run-cancer percentiles equally within metric; ascending mean "
        "percentile defines consensus rank."
    )
    result["cutoff_status"] = "fixed_for_revision_not_prospectively_preregistered"
    result.to_csv(OUTPUT / "formal_hdac_enrichment.csv", index=False)
    return result


def map_brd_to_library() -> pd.DataFrame:
    consensus = pd.read_csv(
        RESULTS / "screening_analysis" / "compound_consensus_signed_wtrs.csv",
        low_memory=False,
    )
    rows: list[dict[str, object]] = []
    for row in consensus.itertuples(index=False):
        identifiers = set(
            re.findall(
                r"BRD-[A-Z]\d+",
                f"{getattr(row, 'source_drug_ids', '')} {getattr(row, 'compoundinfo_pert_id', '')}",
            )
        )
        for identifier in identifiers:
            rows.append(
                {
                    "brd_id": identifier,
                    "compound_index": row.compound_index,
                    "canonical_smiles": row.canonical_smiles,
                    "display_name": row.display_name,
                    "compoundinfo_name": row.compoundinfo_name,
                    "target": row.target,
                    "moa": row.moa,
                    "hdac_annotation_source": row.hdac_annotation_source,
                    "is_hdac_annotated": row.is_hdac_annotated,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("brd_id")


def hdac_holdout_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_path = (
        RESULTS
        / "benchmarks"
        / "hdac_class_holdout"
        / "dual_stream"
        / "seed_1"
        / "test_profile_metrics.csv"
    )
    prefix = RESULTS / "benchmarks" / "hdac_class_holdout_pre_fix_20260717_160350"
    pre_path = prefix / "dual_stream" / "seed_1" / "test_profile_metrics.csv"
    library_map = map_brd_to_library()

    def summarize(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, usecols=["sig_id"])
        frame["brd_id"] = frame["sig_id"].str.extract(r"(BRD-[A-Z]\d+)")[0]
        frame["cache_cell_label"] = frame["sig_id"].str.extract(r"^[^_]+_([^_]+)_")[0]
        return (
            frame.groupby("brd_id", as_index=False)
            .agg(test_signatures=("sig_id", "size"), test_cache_cell_lines=("cache_cell_label", "nunique"))
            .merge(library_map, on="brd_id", how="left", validate="one_to_one")
        )

    corrected = summarize(corrected_path)
    pre = summarize(pre_path)[["brd_id", "test_signatures"]].rename(
        columns={"test_signatures": "pre_fix_test_signatures"}
    )
    corrected = corrected.merge(pre, on="brd_id", how="left")
    corrected["pre_fix_test_signatures"] = corrected["pre_fix_test_signatures"].fillna(0).astype(int)
    corrected["added_by_annotation_fix"] = corrected["pre_fix_test_signatures"].eq(0)
    corrected["nch51_removed_from_training"] = corrected["display_name"].eq("NCH-51")
    corrected = corrected.sort_values(["display_name", "brd_id"], na_position="last")
    if len(corrected) != 30 or corrected["canonical_smiles"].nunique() != 30:
        raise RuntimeError("Corrected holdout inventory is not exactly 30 structures")
    corrected.to_csv(OUTPUT / "corrected_hdac_holdout_30_structures.csv", index=False)

    def overview(label: str, path: Path) -> dict[str, object]:
        data = pd.read_csv(path, usecols=["sig_id"])
        data["brd_id"] = data["sig_id"].str.extract(r"(BRD-[A-Z]\d+)")[0]
        data["cache_cell_label"] = data["sig_id"].str.extract(r"^[^_]+_([^_]+)_")[0]
        return {
            "version": label,
            "test_signatures": len(data),
            "test_structures": int(data["brd_id"].nunique()),
            "test_cache_cell_lines": int(data["cache_cell_label"].nunique()),
        }

    comparison = pd.DataFrame(
        [overview("discarded_pre_fix", pre_path), overview("corrected_final", corrected_path)]
    )
    comparison["signature_change_vs_pre_fix"] = comparison["test_signatures"] - int(
        comparison.loc[comparison["version"] == "discarded_pre_fix", "test_signatures"].iloc[0]
    )
    comparison["structure_change_vs_pre_fix"] = comparison["test_structures"] - int(
        comparison.loc[comparison["version"] == "discarded_pre_fix", "test_structures"].iloc[0]
    )
    comparison.to_csv(OUTPUT / "corrected_hdac_holdout_pre_post_fix.csv", index=False)
    return corrected, comparison


def measured_background_rank_sensitivity() -> pd.DataFrame:
    ranks = pd.read_csv(RESULTS / "measured_reversal" / "candidate_measured_compound_ranks.csv")
    ranks = ranks.loc[ranks["candidate"].isin(PRIMARY_DISPLAY)].copy()
    rows: list[dict[str, object]] = []
    for (candidate, metric), subset in ranks.groupby(["candidate", "metric"]):
        percentile = subset["measured_compound_rank_percentile"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "metric": metric,
                "cancers": int(subset["cancer"].nunique()),
                "all_measured_canonical_compounds": int(subset["measured_compounds"].iloc[0]),
                "median_rank_percentile_lower_is_better": float(np.median(percentile)),
                "q25_rank_percentile": float(np.quantile(percentile, 0.25)),
                "q75_rank_percentile": float(np.quantile(percentile, 0.75)),
                "fraction_within_top_10_percent": float(np.mean(percentile <= 0.10)),
                "evidence_scope": "all_profile_unmatched_sensitivity_not_condition_matched_null",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "measured_all_profile_background_rank_sensitivity.csv", index=False)
    return result


def plot_screening_landscape(enrichment: pd.DataFrame) -> None:
    predicted = pd.read_csv(
        RESULTS / "measured_reversal" / "corrected_hdac_predicted_candidate_scores.csv"
    )
    predicted = predicted.loc[
        (predicted["metric"] == "signed_wtrs") & predicted["candidate"].isin(PRIMARY_DISPLAY)
    ]
    matrix = (
        predicted.groupby(["candidate", "cancer"])["predicted_score"]
        .mean()
        .unstack("cancer")
        .loc[PRIMARY_DISPLAY]
    )
    cancers = matrix.columns.tolist()

    fig = plt.figure(figsize=(15.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.6, 1.0])
    ax_heat = fig.add_subplot(grid[0, :])
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = ax_heat.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax_heat.set_xticks(range(len(cancers)), cancers, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(len(PRIMARY_DISPLAY)), PRIMARY_DISPLAY, fontsize=9)
    ax_heat.set_title(
        "A  Corrected-HDAC ensemble predicted signed reversal across frozen candidates and cancers",
        loc="left",
        fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.018, pad=0.012)
    colorbar.set_label("Predicted signed wTRS (larger = stronger opposition)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax_heat.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5)

    ax_dist = fig.add_subplot(grid[1, 0])
    values = [matrix.loc[candidate].to_numpy(dtype=float) for candidate in PRIMARY_DISPLAY]
    box = ax_dist.boxplot(values, patch_artist=True, widths=0.6, showfliers=False)
    role_lookup = {
        "Mocetinostat": "core_candidate",
        "NCH-51": "secondary_candidate",
        "TC-H-106": "original_method_sensitive",
        "Belinostat": "class_support",
        "PCI-24781": "class_support",
        "Panobinostat": "class_support",
        "Entinostat": "reference_control",
        "Vorinostat": "reference_control",
    }
    for patch, candidate in zip(box["boxes"], PRIMARY_DISPLAY):
        patch.set_facecolor(ROLE_COLORS[role_lookup[candidate]])
        patch.set_alpha(0.72)
    jitter_rng = np.random.default_rng(SEED)
    for index, candidate_values in enumerate(values, start=1):
        ax_dist.scatter(
            jitter_rng.normal(index, 0.055, len(candidate_values)),
            candidate_values,
            s=12,
            color="#222222",
            alpha=0.45,
            linewidths=0,
        )
    ax_dist.axhline(0, color="#555555", lw=0.8, ls="--")
    ax_dist.set_xticks(range(1, len(PRIMARY_DISPLAY) + 1), PRIMARY_DISPLAY, rotation=35, ha="right")
    ax_dist.set_ylabel("Predicted signed wTRS")
    ax_dist.set_title(
        "B  Descriptive across-cancer distributions (22 cancer signatures per compound)",
        loc="left",
        fontweight="bold",
    )
    ax_dist.grid(axis="y", color="#dddddd", lw=0.6)

    ax_enrich = fig.add_subplot(grid[1, 1])
    subset = enrichment.loc[
        (enrichment["annotation_definition"] == "explicit_class_I_target")
        & enrichment["metric"].isin(PRIMARY_METRICS)
    ].copy()
    x = np.arange(4)
    width = 0.34
    for offset, metric, color, label in [
        (-width / 2, "signed_wtrs", "#9c2f2f", "Signed wTRS (primary)"),
        (width / 2, "spearman_reversal", "#4f7ea8", "Spearman reversal (co-primary)"),
    ]:
        part = subset.loc[subset["metric"] == metric].sort_values("top_fraction")
        bars = ax_enrich.bar(
            x + offset,
            part["fold_enrichment"],
            width=width,
            color=color,
            alpha=0.82,
            label=label,
        )
        for bar, q in zip(bars, part["fdr_bh_within_metric_definition"]):
            marker = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
            ax_enrich.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                marker,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_enrich.axhline(1, color="#555555", lw=0.8, ls="--")
    ax_enrich.set_xticks(x, ["Top 0.5%", "Top 1%", "Top 5%", "Top 10%"])
    ax_enrich.set_ylabel("Class-I-target fold enrichment")
    ax_enrich.set_title("C  Fixed revision top-fraction enrichment", loc="left", fontweight="bold")
    ax_enrich.legend(frameon=False, fontsize=8)
    ax_enrich.grid(axis="y", color="#dddddd", lw=0.6)
    ax_enrich.text(
        0,
        -0.23,
        "Fisher exact tests; BH-FDR within metric/definition. * q<0.05, ** q<0.01, *** q<0.001.",
        transform=ax_enrich.transAxes,
        fontsize=7.5,
    )

    fig.suptitle(
        "Frozen pan-cancer screening landscape and formal HDAC enrichment",
        fontsize=16,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"screening_landscape_hdac_enrichment.{suffix}", dpi=300)
    plt.close(fig)


def plot_primary_stability(source: pd.DataFrame, comparison: pd.DataFrame) -> None:
    primary_source = source.loc[source["metric"].isin(PRIMARY_METRICS)].copy()
    order = [candidate for candidate in CANDIDATE_ORDER if candidate in set(primary_source["candidate"])]
    roles = primary_source.drop_duplicates("candidate").set_index("candidate")["candidate_role"].to_dict()
    fig, (ax_box, ax_compare) = plt.subplots(
        1, 2, figsize=(15.5, 6.8), gridspec_kw={"width_ratios": [1.6, 1.0]}, constrained_layout=True
    )
    values = [
        primary_source.loc[primary_source["candidate"] == candidate, "median_library_percentile"].to_numpy()
        for candidate in order
    ]
    box = ax_box.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
    for patch, candidate in zip(box["boxes"], order):
        patch.set_facecolor(ROLE_COLORS.get(roles[candidate], "#888888"))
        patch.set_alpha(0.75)
    ax_box.set_xticks(range(1, len(order) + 1), [x.replace("_or_", "/") for x in order], rotation=38, ha="right")
    ax_box.set_ylim(-2, 102)
    ax_box.set_ylabel("Library percentile (higher = stronger rank)")
    ax_box.set_title("A  Primary 48-config stability: signed wTRS + Spearman reversal", loc="left", fontweight="bold")
    ax_box.grid(axis="y", color="#dddddd", lw=0.6)

    aligned = comparison.set_index("candidate").loc[order]
    y = np.arange(len(order))
    x48 = aligned["median_run_library_percentile_primary48"].to_numpy()
    x72 = aligned["median_run_library_percentile_sensitivity72"].to_numpy()
    for yi, left, right in zip(y, x72, x48):
        ax_compare.plot([left, right], [yi, yi], color="#bbbbbb", lw=1.5)
    ax_compare.scatter(x72, y, color="#888888", marker="o", label="72-config incl. legacy", zorder=3)
    ax_compare.scatter(x48, y, color="#9c2f2f", marker="D", label="48-config primary", zorder=3)
    ax_compare.set_yticks(y, [x.replace("_or_", "/") for x in order])
    ax_compare.invert_yaxis()
    ax_compare.set_xlim(45, 100)
    ax_compare.set_xlabel("Median library percentile")
    ax_compare.set_title("B  Primary versus legacy-inclusive summary", loc="left", fontweight="bold")
    ax_compare.legend(frameon=False, fontsize=8)
    ax_compare.grid(axis="x", color="#dddddd", lw=0.6)
    fig.suptitle("Candidate stability after metric hierarchy correction", fontsize=16, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"candidate_stability_primary.{suffix}", dpi=300)
    plt.close(fig)


def plot_predicted_measured_two_way(correlations: pd.DataFrame) -> None:
    data = pd.read_csv(
        RESULTS / "measured_lincs_figures" / "predicted_measured_hiq_plot_data.csv"
    )
    candidate_colors = dict(zip(PRIMARY_DISPLAY, plt.cm.tab10.colors[:8]))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.4), constrained_layout=True)
    for row, metric in enumerate(PRIMARY_METRICS):
        for col, model in enumerate(MODEL_LABELS):
            ax = axes[row, col]
            subset = data.loc[(data["metric"] == metric) & (data["model"] == model)]
            for candidate in PRIMARY_DISPLAY:
                part = subset.loc[subset["candidate"] == candidate]
                ax.scatter(
                    part["measured_cell_score_median"],
                    part["predicted_score_mean"],
                    s=25,
                    alpha=0.72,
                    color=candidate_colors[candidate],
                    label=candidate if row == 0 and col == 1 else None,
                    edgecolor="white",
                    linewidth=0.25,
                )
            x = subset["measured_cell_score_median"].to_numpy(dtype=float)
            y = subset["predicted_score_mean"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(line_x, slope * line_x + intercept, color="#222222", lw=1.1)
            stat = correlations.loc[
                (correlations["metric"] == metric) & (correlations["model"] == model)
            ].iloc[0]
            ax.text(
                0.03,
                0.97,
                (
                    f"Pearson r={stat.pearson:.3f} "
                    f"[{stat.pearson_two_way_ci95_low:.3f}, {stat.pearson_two_way_ci95_high:.3f}]\n"
                    f"Spearman ρ={stat.spearman:.3f} "
                    f"[{stat.spearman_two_way_ci95_low:.3f}, {stat.spearman_two_way_ci95_high:.3f}]"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
            )
            ax.set_title(f"{MODEL_LABELS[model]} | {metric.replace('_', ' ')}", fontweight="bold")
            ax.set_xlabel("Official HiQ measured reversal")
            ax.set_ylabel("Corrected-HDAC ensemble predicted reversal")
            ax.grid(color="#e5e5e5", lw=0.6)
    axes[0, 1].legend(
        title="Compound",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Predicted versus official-condition measured reversal\n"
        "95% intervals use crossed candidate x cancer bootstrap",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"predicted_measured_hiq_two_way.{suffix}", dpi=300)
    plt.close(fig)


def write_manifest(outputs: list[Path]) -> None:
    manifest = {
        "analysis": "final_revision_cpu_postprocessing",
        "random_seed": SEED,
        "bootstrap_iterations": BOOTSTRAPS,
        "model_retraining": False,
        "candidate_roles_changed": False,
        "background_gene_interpretation": {
            "submitted_custom_entrez_ids": 11144,
            "gprofiler_effective_domain_size": 11154,
            "note": (
                "The first value is the submitted measured background identifier count; "
                "the second is the service-reported mapped statistical universe."
            ),
        },
        "matched_non_hdac_null": {
            "status": "not_computed",
            "reason": (
                "The archived official-condition outputs do not contain a condition-matched "
                "non-HDAC profile universe. The exported all-profile compound ranks are retained "
                "only as an unmatched sensitivity analysis."
            ),
        },
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    with (OUTPUT / "final_revision_postprocessing_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs, generalization, paired = corrected_generalization()
    correlations, loo = two_way_bootstrap()
    stability_source, stability_primary, stability_comparison = primary_stability()
    enrichment = formal_hdac_enrichment()
    holdout, holdout_comparison = hdac_holdout_inventory()
    background_rank = measured_background_rank_sensitivity()
    plot_screening_landscape(enrichment)
    plot_primary_stability(stability_source, stability_comparison)
    plot_predicted_measured_two_way(correlations)
    outputs = [
        OUTPUT / "generalization_all_runs_corrected.csv",
        OUTPUT / "generalization_summary_corrected.csv",
        OUTPUT / "generalization_paired_corrected.csv",
        OUTPUT / "predicted_measured_two_way_bootstrap.csv",
        OUTPUT / "predicted_measured_leave_one_out.csv",
        OUTPUT / "candidate_stability_primary_48.csv",
        OUTPUT / "candidate_stability_legacy_inclusive_72.csv",
        OUTPUT / "candidate_stability_primary_vs_sensitivity.csv",
        OUTPUT / "formal_hdac_enrichment.csv",
        OUTPUT / "corrected_hdac_holdout_30_structures.csv",
        OUTPUT / "corrected_hdac_holdout_pre_post_fix.csv",
        OUTPUT / "measured_all_profile_background_rank_sensitivity.csv",
        FIGURES / "screening_landscape_hdac_enrichment.pdf",
        FIGURES / "candidate_stability_primary.pdf",
        FIGURES / "predicted_measured_hiq_two_way.pdf",
    ]
    write_manifest(outputs)
    print("FINAL REVISION CPU POST-PROCESSING: PASSED")
    print(f"runs={len(runs)}")
    print(f"two_way_rows={len(correlations)}")
    print(f"primary_stability_rows={len(stability_primary)}")
    print(f"formal_enrichment_rows={len(enrichment)}")
    print(f"corrected_hdac_structures={len(holdout)}")
    print(f"background_rank_rows={len(background_rank)}")


if __name__ == "__main__":
    main()
