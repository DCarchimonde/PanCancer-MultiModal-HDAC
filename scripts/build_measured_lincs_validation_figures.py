from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr


CANDIDATE_ORDER = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
    "Tianeptinaline_or_BG-1010",
]

STRICT_CANDIDATES = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
]

CANDIDATE_ROLES = {
    "Mocetinostat": "core_candidate",
    "NCH-51": "secondary_candidate",
    "TC-H-106": "original_method_sensitive",
    "Belinostat": "class_support",
    "PCI-24781": "class_support",
    "Panobinostat": "class_support",
    "Entinostat": "reference_control",
    "Vorinostat": "reference_control",
    "Tianeptinaline_or_BG-1010": "identity_conflict_excluded",
}

METRICS = ["legacy_wtrs", "signed_wtrs", "spearman_reversal"]
PRIMARY_METRICS = ["signed_wtrs", "spearman_reversal"]
QUALITY_ORDER = ["all_profiles", "qc_pass", "hiq"]
QUALITY_LABELS = {
    "all_profiles": "All profiles",
    "qc_pass": "QC pass",
    "hiq": "HiQ",
}
QUALITY_COLORS = {
    "all_profiles": "#7f7f7f",
    "qc_pass": "#ef8a62",
    "hiq": "#2166ac",
}
MODEL_LABELS = {
    "dual_stream": "Dual-stream",
    "fingerprint_mlp": "Fingerprint MLP",
}
CANDIDATE_COLORS = {
    candidate: plt.get_cmap("tab10")(index)
    for index, candidate in enumerate(STRICT_CANDIDATES)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reaggregate measured LINCS reversal with official conditions and "
            "build measured-reversal and corrected-HDAC prediction-validation figures."
        )
    )
    parser.add_argument(
        "--measured-root",
        type=Path,
        default=Path("results/revision/measured_reversal"),
    )
    parser.add_argument(
        "--condition-root",
        type=Path,
        default=Path("results/revision/lincs_condition_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/measured_lincs_figures"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=20260718)
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "yes", "y"})
    )


def normalized_unit(unit: object) -> str:
    return (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(" ", "")
    )


def convert_values(
    values: pd.Series,
    units: pd.Series,
    factors: dict[str, float],
    label: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized_units = units.map(normalized_unit)
    unknown = sorted(
        set(normalized_units.loc[numeric.notna()]) - set(factors)
    )
    if unknown:
        raise RuntimeError(f"Unsupported official {label} units: {unknown}")
    factor_values = normalized_units.map(factors)
    converted = numeric * factor_values
    if converted.isna().any():
        missing = int(converted.isna().sum())
        raise RuntimeError(f"Could not normalize {missing} official {label} rows")
    return converted.astype(float)


def load_official_profile_scores(
    score_path: Path,
    condition_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(score_path, low_memory=False)
    conditions = pd.read_csv(condition_path, low_memory=False)
    require_columns(
        scores,
        ["sig_id", "candidate", "cancer", "metric", "score"],
        "measured signature scores",
    )
    require_columns(
        conditions,
        [
            "sig_id",
            "candidate",
            "cache_cell_id",
            "official_cell_iname",
            "official_pert_dose",
            "official_pert_dose_unit",
            "official_pert_time",
            "official_pert_time_unit",
            "official_nsample",
            "official_tas",
            "official_is_hiq",
            "official_qc_pass",
        ],
        "official candidate conditions",
    )
    if conditions["sig_id"].duplicated().any():
        raise RuntimeError("Official candidate conditions contain duplicate sig_id")
    expected_candidates = set(CANDIDATE_ORDER)
    score_candidates = set(scores["candidate"].astype(str).unique())
    condition_candidates = set(conditions["candidate"].dropna().astype(str).unique())
    if score_candidates != expected_candidates or condition_candidates != expected_candidates:
        raise RuntimeError(
            "Measured candidate set changed: "
            f"scores={sorted(score_candidates)}, "
            f"conditions={sorted(condition_candidates)}"
        )

    condition_columns = [
        "sig_id",
        "candidate",
        "cache_cell_id",
        "official_cell_iname",
        "official_pert_dose",
        "official_pert_dose_unit",
        "official_pert_time",
        "official_pert_time_unit",
        "official_nsample",
        "official_tas",
        "official_is_hiq",
        "official_qc_pass",
    ]
    conditions = conditions[condition_columns].rename(
        columns={"candidate": "official_candidate"}
    )
    profile_scores = scores.merge(
        conditions,
        on="sig_id",
        how="left",
        validate="many_to_one",
    )
    if profile_scores["official_candidate"].isna().any():
        raise RuntimeError("At least one measured score lacks official conditions")
    mismatch = profile_scores["candidate"].ne(
        profile_scores["official_candidate"]
    )
    if mismatch.any():
        raise RuntimeError(
            f"Candidate labels disagree for {int(mismatch.sum())} score rows"
        )
    profile_scores = profile_scores.drop(columns=["official_candidate"])
    profile_scores["candidate_role"] = profile_scores["candidate"].map(
        CANDIDATE_ROLES
    )
    profile_scores["qc_pass"] = boolean_series(
        profile_scores["official_qc_pass"]
    )
    profile_scores["hiq"] = boolean_series(
        profile_scores["official_is_hiq"]
    )
    profile_scores["time_h"] = convert_values(
        profile_scores["official_pert_time"],
        profile_scores["official_pert_time_unit"],
        {
            "h": 1.0,
            "hr": 1.0,
            "hrs": 1.0,
            "hour": 1.0,
            "hours": 1.0,
            "min": 1.0 / 60.0,
            "mins": 1.0 / 60.0,
            "minute": 1.0 / 60.0,
            "minutes": 1.0 / 60.0,
            "d": 24.0,
            "day": 24.0,
            "days": 24.0,
        },
        "time",
    )
    profile_scores["dose_um"] = convert_values(
        profile_scores["official_pert_dose"],
        profile_scores["official_pert_dose_unit"],
        {
            "um": 1.0,
            "micromolar": 1.0,
            "nm": 0.001,
            "nanomolar": 0.001,
            "mm": 1_000.0,
            "millimolar": 1_000.0,
            "pm": 0.000001,
            "picomolar": 0.000001,
        },
        "dose",
    )
    profile_scores["official_nsample"] = pd.to_numeric(
        profile_scores["official_nsample"], errors="coerce"
    )
    profile_scores["official_tas"] = pd.to_numeric(
        profile_scores["official_tas"], errors="coerce"
    )
    if profile_scores[["official_nsample", "official_tas"]].isna().any().any():
        raise RuntimeError("Official nsample/TAS normalization produced missing values")

    score_counts = profile_scores.groupby("sig_id").size()
    expected_per_signature = (
        profile_scores["cancer"].nunique()
        * profile_scores["metric"].nunique()
    )
    if not score_counts.eq(expected_per_signature).all():
        raise RuntimeError("Measured score rows are incomplete for some signatures")

    unique_conditions = profile_scores.drop_duplicates("sig_id").copy()
    cell_differences = unique_conditions.loc[
        unique_conditions["cache_cell_id"].astype(str)
        != unique_conditions["official_cell_iname"].astype(str)
    ].copy()
    difference_summary = (
        cell_differences.groupby("candidate", as_index=False)
        .agg(
            affected_signatures=("sig_id", "nunique"),
            cache_cell_labels=("cache_cell_id", "nunique"),
            official_cell_labels=("official_cell_iname", "nunique"),
        )
        if len(cell_differences)
        else pd.DataFrame(
            columns=[
                "candidate",
                "affected_signatures",
                "cache_cell_labels",
                "official_cell_labels",
            ]
        )
    )
    return profile_scores, unique_conditions, difference_summary


def expand_quality_sets(profile_scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    all_profiles = profile_scores.copy()
    all_profiles["quality_set"] = "all_profiles"
    frames.append(all_profiles)
    qc_pass = profile_scores.loc[profile_scores["qc_pass"]].copy()
    qc_pass["quality_set"] = "qc_pass"
    frames.append(qc_pass)
    hiq = profile_scores.loc[profile_scores["hiq"]].copy()
    hiq["quality_set"] = "hiq"
    frames.append(hiq)
    expanded = pd.concat(frames, ignore_index=True)
    if set(expanded["quality_set"].unique()) != set(QUALITY_ORDER):
        raise RuntimeError("At least one required quality stratum is empty")
    return expanded


def summarize_measured_scores(
    expanded: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    condition_scores = (
        expanded.groupby(
            [
                "candidate",
                "candidate_role",
                "quality_set",
                "metric",
                "cancer",
                "official_cell_iname",
                "time_h",
                "dose_um",
            ],
            as_index=False,
        )
        .agg(
            condition_median_score=("score", "median"),
            signatures=("sig_id", "nunique"),
            nsample_sum=("official_nsample", "sum"),
            tas_median=("official_tas", "median"),
        )
    )
    cell_scores = (
        condition_scores.groupby(
            [
                "candidate",
                "candidate_role",
                "quality_set",
                "metric",
                "cancer",
                "official_cell_iname",
            ],
            as_index=False,
        )
        .agg(
            cell_median_score=("condition_median_score", "median"),
            conditions=("condition_median_score", "size"),
            signatures=("signatures", "sum"),
        )
    )
    cancer_summary = (
        cell_scores.groupby(
            [
                "candidate",
                "candidate_role",
                "quality_set",
                "metric",
                "cancer",
            ],
            as_index=False,
        )
        .agg(
            official_cell_lines=("official_cell_iname", "nunique"),
            conditions=("conditions", "sum"),
            signatures=("signatures", "sum"),
            cell_score_mean=("cell_median_score", "mean"),
            cell_score_median=("cell_median_score", "median"),
            cell_score_q25=("cell_median_score", lambda x: x.quantile(0.25)),
            cell_score_q75=("cell_median_score", lambda x: x.quantile(0.75)),
            positive_cell_fraction=("cell_median_score", lambda x: (x > 0).mean()),
        )
    )
    pan_cancer_summary = (
        cancer_summary.groupby(
            ["candidate", "candidate_role", "quality_set", "metric"],
            as_index=False,
        )
        .agg(
            cancers=("cancer", "nunique"),
            signatures=("signatures", "max"),
            official_cell_lines=("official_cell_lines", "max"),
            median_across_cancers=("cell_score_median", "median"),
            mean_across_cancers=("cell_score_median", "mean"),
            positive_cancer_fraction=("cell_score_median", lambda x: (x > 0).mean()),
            cancer_score_q25=("cell_score_median", lambda x: x.quantile(0.25)),
            cancer_score_q75=("cell_score_median", lambda x: x.quantile(0.75)),
        )
    )

    cell_time = (
        condition_scores.groupby(
            [
                "candidate",
                "candidate_role",
                "quality_set",
                "metric",
                "cancer",
                "official_cell_iname",
                "time_h",
            ],
            as_index=False,
        )
        .agg(
            cell_time_median_score=("condition_median_score", "median"),
            doses=("dose_um", "nunique"),
            signatures=("signatures", "sum"),
        )
    )
    time_summary = (
        cell_time.groupby(
            ["candidate", "candidate_role", "quality_set", "metric", "time_h"],
            as_index=False,
        )
        .agg(
            cell_cancer_units=("cell_time_median_score", "size"),
            cancers=("cancer", "nunique"),
            official_cell_lines=("official_cell_iname", "nunique"),
            signature_cancer_observations=("signatures", "sum"),
            median_score=("cell_time_median_score", "median"),
            mean_score=("cell_time_median_score", "mean"),
            q25_score=("cell_time_median_score", lambda x: x.quantile(0.25)),
            q75_score=("cell_time_median_score", lambda x: x.quantile(0.75)),
        )
    )
    time_counts = (
        expanded.groupby(
            ["candidate", "quality_set", "metric", "time_h"],
            as_index=False,
        )["sig_id"]
        .nunique()
        .rename(columns={"sig_id": "signatures"})
    )
    time_summary = time_summary.merge(
        time_counts,
        on=["candidate", "quality_set", "metric", "time_h"],
        how="left",
        validate="one_to_one",
    )

    cell_dose = (
        condition_scores.groupby(
            [
                "candidate",
                "candidate_role",
                "quality_set",
                "metric",
                "cancer",
                "official_cell_iname",
                "dose_um",
            ],
            as_index=False,
        )
        .agg(
            cell_dose_median_score=("condition_median_score", "median"),
            times=("time_h", "nunique"),
            signatures=("signatures", "sum"),
        )
    )
    dose_summary = (
        cell_dose.groupby(
            ["candidate", "candidate_role", "quality_set", "metric", "dose_um"],
            as_index=False,
        )
        .agg(
            cell_cancer_units=("cell_dose_median_score", "size"),
            cancers=("cancer", "nunique"),
            official_cell_lines=("official_cell_iname", "nunique"),
            signature_cancer_observations=("signatures", "sum"),
            median_score=("cell_dose_median_score", "median"),
            mean_score=("cell_dose_median_score", "mean"),
            q25_score=("cell_dose_median_score", lambda x: x.quantile(0.25)),
            q75_score=("cell_dose_median_score", lambda x: x.quantile(0.75)),
        )
    )
    dose_counts = (
        expanded.groupby(
            ["candidate", "quality_set", "metric", "dose_um"],
            as_index=False,
        )["sig_id"]
        .nunique()
        .rename(columns={"sig_id": "signatures"})
    )
    dose_summary = dose_summary.merge(
        dose_counts,
        on=["candidate", "quality_set", "metric", "dose_um"],
        how="left",
        validate="one_to_one",
    )
    return (
        condition_scores,
        cell_scores,
        cancer_summary,
        pan_cancer_summary,
        time_summary,
        dose_summary,
    )


def quality_coverage(unique_conditions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_ORDER:
        subset = unique_conditions.loc[
            unique_conditions["candidate"] == candidate
        ]
        rows.append(
            {
                "candidate": candidate,
                "candidate_role": CANDIDATE_ROLES[candidate],
                "all_profiles": int(subset["sig_id"].nunique()),
                "qc_pass_profiles": int(
                    subset.loc[subset["qc_pass"], "sig_id"].nunique()
                ),
                "hiq_profiles": int(
                    subset.loc[subset["hiq"], "sig_id"].nunique()
                ),
                "official_cell_lines": int(
                    subset["official_cell_iname"].nunique()
                ),
                "times_h": " | ".join(
                    f"{value:g}"
                    for value in sorted(subset["time_h"].unique())
                ),
                "dose_levels": int(subset["dose_um"].nunique()),
                "dose_um_min": float(subset["dose_um"].min()),
                "dose_um_max": float(subset["dose_um"].max()),
            }
        )
    return pd.DataFrame(rows)


def safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return (
        float(pearsonr(x, y).statistic),
        float(spearmanr(x, y).statistic),
    )


def cluster_bootstrap_correlations(
    frame: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    candidates = frame["candidate"].unique()
    pearson_values = []
    spearman_values = []
    for _ in range(iterations):
        sampled = rng.choice(candidates, size=len(candidates), replace=True)
        bootstrap = pd.concat(
            [frame.loc[frame["candidate"] == candidate] for candidate in sampled],
            ignore_index=True,
        )
        pearson, spearman = safe_correlations(
            bootstrap["predicted_score_mean"].to_numpy(dtype=float),
            bootstrap["measured_cell_score_median"].to_numpy(dtype=float),
        )
        pearson_values.append(pearson)
        spearman_values.append(spearman)
    pearson_array = np.asarray(pearson_values, dtype=float)
    spearman_array = np.asarray(spearman_values, dtype=float)
    return {
        "pearson_ci95_low": float(np.nanpercentile(pearson_array, 2.5)),
        "pearson_ci95_high": float(np.nanpercentile(pearson_array, 97.5)),
        "spearman_ci95_low": float(np.nanpercentile(spearman_array, 2.5)),
        "spearman_ci95_high": float(np.nanpercentile(spearman_array, 97.5)),
    }


def prediction_measured_tables(
    predicted_path: Path,
    cancer_summary: pd.DataFrame,
    iterations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predicted = pd.read_csv(predicted_path, low_memory=False)
    require_columns(
        predicted,
        [
            "candidate",
            "cancer",
            "metric",
            "model",
            "seed",
            "predicted_score",
        ],
        "corrected-HDAC predicted candidate scores",
    )
    measured_hiq = cancer_summary.loc[
        (cancer_summary["quality_set"] == "hiq")
        & cancer_summary["candidate"].isin(STRICT_CANDIDATES),
        [
            "candidate",
            "cancer",
            "metric",
            "official_cell_lines",
            "conditions",
            "signatures",
            "cell_score_median",
        ],
    ].rename(
        columns={"cell_score_median": "measured_cell_score_median"}
    )
    strict_predictions = predicted.loc[
        predicted["candidate"].isin(STRICT_CANDIDATES)
    ].copy()
    by_seed = strict_predictions.merge(
        measured_hiq,
        on=["candidate", "cancer", "metric"],
        how="inner",
        validate="many_to_one",
    )
    expected_seed_rows = (
        len(STRICT_CANDIDATES)
        * measured_hiq["cancer"].nunique()
        * len(METRICS)
        * strict_predictions["model"].nunique()
        * strict_predictions["seed"].nunique()
    )
    if len(by_seed) != expected_seed_rows:
        raise RuntimeError(
            f"Prediction/measured seed rows {len(by_seed)} != {expected_seed_rows}"
        )

    seed_correlation_rows = []
    for (metric, model, seed), subset in by_seed.groupby(
        ["metric", "model", "seed"], sort=True
    ):
        pearson, spearman = safe_correlations(
            subset["predicted_score"].to_numpy(dtype=float),
            subset["measured_cell_score_median"].to_numpy(dtype=float),
        )
        seed_correlation_rows.append(
            {
                "metric": metric,
                "model": model,
                "seed": int(seed),
                "strict_candidates": int(subset["candidate"].nunique()),
                "candidate_cancer_pairs": len(subset),
                "pearson": pearson,
                "spearman": spearman,
            }
        )
    by_seed_correlations = pd.DataFrame(seed_correlation_rows)

    prediction_ensemble = (
        strict_predictions.groupby(
            ["candidate", "cancer", "metric", "model"], as_index=False
        )
        .agg(
            predicted_score_mean=("predicted_score", "mean"),
            predicted_score_seed_sd=("predicted_score", "std"),
            seeds=("seed", "nunique"),
        )
    )
    plot_data = prediction_ensemble.merge(
        measured_hiq,
        on=["candidate", "cancer", "metric"],
        how="inner",
        validate="many_to_one",
    )
    rng = np.random.default_rng(random_seed)
    correlation_rows = []
    for (metric, model), subset in plot_data.groupby(
        ["metric", "model"], sort=True
    ):
        pearson, spearman = safe_correlations(
            subset["predicted_score_mean"].to_numpy(dtype=float),
            subset["measured_cell_score_median"].to_numpy(dtype=float),
        )
        intervals = cluster_bootstrap_correlations(subset, iterations, rng)
        correlation_rows.append(
            {
                "metric": metric,
                "model": model,
                "seeds": int(subset["seeds"].min()),
                "strict_candidates": int(subset["candidate"].nunique()),
                "candidate_cancer_pairs": len(subset),
                "pearson": pearson,
                "spearman": spearman,
                **intervals,
                "bootstrap_unit": "candidate",
                "bootstrap_iterations": iterations,
            }
        )
    return plot_data, pd.DataFrame(correlation_rows), by_seed_correlations


def measured_reversal_figure(
    cancer_summary: pd.DataFrame,
    pan_cancer_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    heatmap_data = cancer_summary.loc[
        (cancer_summary["quality_set"] == "hiq")
        & (cancer_summary["metric"] == "spearman_reversal")
        & cancer_summary["candidate"].isin(STRICT_CANDIDATES)
    ]
    cancer_order = heatmap_data["cancer"].drop_duplicates().tolist()
    heatmap = heatmap_data.pivot(
        index="candidate", columns="cancer", values="cell_score_median"
    ).reindex(index=STRICT_CANDIDATES, columns=cancer_order)
    if heatmap.isna().any().any():
        raise RuntimeError("HiQ measured-reversal heatmap contains missing cells")

    figure = plt.figure(figsize=(22, 12), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    heat_axis = figure.add_subplot(grid[0, :])
    heat_values = heatmap.to_numpy(dtype=float)
    limit = max(float(np.max(np.abs(heat_values))), 1e-6)
    image = heat_axis.imshow(
        heat_values,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    heat_axis.set_xticks(np.arange(len(cancer_order)))
    heat_axis.set_xticklabels(cancer_order, rotation=45, ha="right", fontsize=9)
    heat_axis.set_yticks(np.arange(len(STRICT_CANDIDATES)))
    heat_axis.set_yticklabels(STRICT_CANDIDATES, fontsize=10)
    heat_axis.set_title(
        "A  HiQ measured Spearman reversal, aggregated condition → official cell line",
        loc="left",
        fontweight="bold",
    )
    for row in range(len(STRICT_CANDIDATES)):
        for column in range(len(cancer_order)):
            value = heat_values[row, column]
            color = "white" if abs(value) > limit * 0.58 else "black"
            heat_axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.018, pad=0.01)
    colorbar.set_label("Measured Spearman reversal (−rho; higher = stronger)")

    for axis_index, metric in enumerate(PRIMARY_METRICS):
        axis = figure.add_subplot(grid[1, axis_index])
        subset = pan_cancer_summary.loc[
            (pan_cancer_summary["metric"] == metric)
            & pan_cancer_summary["candidate"].isin(STRICT_CANDIDATES)
        ]
        x = np.arange(len(STRICT_CANDIDATES))
        for quality in QUALITY_ORDER:
            values = (
                subset.loc[subset["quality_set"] == quality]
                .set_index("candidate")
                .reindex(STRICT_CANDIDATES)["median_across_cancers"]
                .to_numpy(dtype=float)
            )
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.6,
                color=QUALITY_COLORS[quality],
                label=QUALITY_LABELS[quality],
            )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xticks(x)
        axis.set_xticklabels(STRICT_CANDIDATES, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        metric_label = (
            "Signed WTRS" if metric == "signed_wtrs" else "Spearman reversal"
        )
        axis.set_ylabel(f"Pan-cancer median {metric_label}")
        axis.set_title(
            f"{'B' if axis_index == 0 else 'C'}  {metric_label}: quality-stratum sensitivity",
            loc="left",
            fontweight="bold",
        )
        if axis_index == 0:
            axis.legend(frameon=False)

    figure.suptitle(
        "Measured LINCS reversal under official dose, time, cell, and QC metadata",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.012,
        (
            "Official siginfo is authoritative. Scores are summarized within "
            "dose/time condition, then within official cell line, then across cell "
            "lines. RG2833 has no measured signature; the identity-conflicted "
            "Tianeptinaline/BG-1010 row is excluded from primary panels."
        ),
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        output_dir / "measured_reversal_official_conditions.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "measured_reversal_official_conditions.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def prediction_measured_figure(
    plot_data: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    models = ["dual_stream", "fingerprint_mlp"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for row, metric in enumerate(PRIMARY_METRICS):
        for column, model in enumerate(models):
            axis = axes[row, column]
            subset = plot_data.loc[
                (plot_data["metric"] == metric)
                & (plot_data["model"] == model)
            ]
            for candidate in STRICT_CANDIDATES:
                candidate_rows = subset.loc[subset["candidate"] == candidate]
                axis.scatter(
                    candidate_rows["predicted_score_mean"],
                    candidate_rows["measured_cell_score_median"],
                    s=28,
                    alpha=0.72,
                    color=CANDIDATE_COLORS[candidate],
                    edgecolors="none",
                    label=candidate,
                )
            x = subset["predicted_score_mean"].to_numpy(dtype=float)
            y = subset["measured_cell_score_median"].to_numpy(dtype=float)
            if len(x) >= 2 and np.std(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(x.min(), x.max(), 100)
                axis.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.2)
            result = correlations.loc[
                (correlations["metric"] == metric)
                & (correlations["model"] == model)
            ].iloc[0]
            axis.text(
                0.03,
                0.97,
                (
                    f"Pearson r = {result['pearson']:.2f} "
                    f"[{result['pearson_ci95_low']:.2f}, {result['pearson_ci95_high']:.2f}]\n"
                    f"Spearman rho = {result['spearman']:.2f} "
                    f"[{result['spearman_ci95_low']:.2f}, {result['spearman_ci95_high']:.2f}]\n"
                    f"n = {int(result['candidate_cancer_pairs'])} candidate–cancer pairs"
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            metric_label = (
                "Signed WTRS" if metric == "signed_wtrs" else "Spearman reversal"
            )
            axis.set_xlabel(f"Predicted {metric_label} (3-seed mean)")
            axis.set_ylabel(f"Measured HiQ {metric_label}")
            panel = chr(ord("A") + row * 2 + column)
            axis.set_title(
                f"{panel}  {MODEL_LABELS[model]} — {metric_label}",
                loc="left",
                fontweight="bold",
            )
            axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center right",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(1.12, 0.5),
    )
    figure.suptitle(
        "Corrected HDAC-holdout predictions versus HiQ measured LINCS reversal",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.065,
        (
            "Predictions are ensembled across seeds 1–3. Measured scores use official "
            "siginfo and condition-first, official-cell-first aggregation. Confidence "
            "intervals use candidate-cluster bootstrap; transcriptomic reversal is "
            "not direct evidence of efficacy or viability response."
        ),
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        output_dir / "corrected_hdac_predicted_vs_measured_hiq.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "corrected_hdac_predicted_vs_measured_hiq.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    score_path = args.measured_root / "candidate_signature_measured_scores.csv"
    predicted_path = (
        args.measured_root / "corrected_hdac_predicted_candidate_scores.csv"
    )
    condition_path = (
        args.condition_root / "official_candidate_conditions.csv"
    )
    for path in [score_path, predicted_path, condition_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    profile_scores, unique_conditions, cell_difference_summary = (
        load_official_profile_scores(score_path, condition_path)
    )
    expanded = expand_quality_sets(profile_scores)
    (
        condition_scores,
        cell_scores,
        cancer_summary,
        pan_cancer_summary,
        time_summary,
        dose_summary,
    ) = summarize_measured_scores(expanded)
    coverage = quality_coverage(unique_conditions)
    plot_data, correlations, by_seed_correlations = prediction_measured_tables(
        predicted_path,
        cancer_summary,
        args.bootstrap_iterations,
        args.random_seed,
    )

    profile_export_columns = [
        "sig_id",
        "candidate",
        "candidate_role",
        "cancer",
        "metric",
        "score",
        "cache_cell_id",
        "official_cell_iname",
        "time_h",
        "dose_um",
        "official_nsample",
        "official_tas",
        "qc_pass",
        "hiq",
    ]
    profile_scores[profile_export_columns].to_csv(
        args.output_dir / "profile_scores_official_conditions.csv", index=False
    )
    condition_scores.to_csv(
        args.output_dir / "condition_level_measured_scores.csv", index=False
    )
    cell_scores.to_csv(
        args.output_dir / "official_cell_level_measured_scores.csv", index=False
    )
    cancer_summary.to_csv(
        args.output_dir / "candidate_cancer_measured_summary_official.csv",
        index=False,
    )
    pan_cancer_summary.to_csv(
        args.output_dir / "candidate_pan_cancer_measured_summary_official.csv",
        index=False,
    )
    time_summary.to_csv(
        args.output_dir / "candidate_time_measured_summary.csv", index=False
    )
    dose_summary.to_csv(
        args.output_dir / "candidate_dose_measured_summary.csv", index=False
    )
    coverage.to_csv(
        args.output_dir / "candidate_condition_quality_coverage.csv", index=False
    )
    cell_difference_summary.to_csv(
        args.output_dir / "candidate_official_cell_label_differences.csv",
        index=False,
    )
    plot_data.to_csv(
        args.output_dir / "predicted_measured_hiq_plot_data.csv", index=False
    )
    correlations.to_csv(
        args.output_dir / "predicted_measured_hiq_correlations.csv", index=False
    )
    by_seed_correlations.to_csv(
        args.output_dir / "predicted_measured_hiq_correlations_by_seed.csv",
        index=False,
    )

    measured_reversal_figure(
        cancer_summary,
        pan_cancer_summary,
        args.output_dir,
        args.figure_dpi,
    )
    prediction_measured_figure(
        plot_data,
        correlations,
        args.output_dir,
        args.figure_dpi,
    )

    runtime = time.perf_counter() - started
    manifest = {
        "measured_signatures": int(unique_conditions["sig_id"].nunique()),
        "measured_candidates": int(unique_conditions["candidate"].nunique()),
        "strict_primary_candidates": STRICT_CANDIDATES,
        "cancers": int(profile_scores["cancer"].nunique()),
        "metrics": METRICS,
        "primary_metrics": PRIMARY_METRICS,
        "quality_sets": QUALITY_ORDER,
        "official_cell_label_difference_signatures": int(
            cell_difference_summary["affected_signatures"].sum()
            if len(cell_difference_summary)
            else 0
        ),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_unit": "candidate",
        "random_seed": args.random_seed,
        "runtime_seconds": runtime,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "measured_score_sha256": sha256_file(score_path),
        "predicted_score_sha256": sha256_file(predicted_path),
        "official_condition_sha256": sha256_file(condition_path),
        "primary_quality_definition": (
            "Official LINCS is_hiq rows; QC-pass and all-profile summaries are "
            "reported as sensitivity analyses."
        ),
        "aggregation_definition": (
            "Median within candidate/dose/time/official-cell condition, then median "
            "across conditions within official cell, then summary across official "
            "cell lines and cancers."
        ),
        "legacy_metric_policy": (
            "legacy_wtrs is retained in exported sensitivity tables but omitted from "
            "primary figures because corrected-HDAC seed stability was weaker."
        ),
        "identity_policy": (
            "RG2833 has no measured LINCS signature. Tianeptinaline/BG-1010 remains "
            "an identity-conflict sensitivity row and is excluded from primary figures."
        ),
        "evidence_note": (
            "Measured transcriptomic reversal is experimental expression evidence, "
            "not direct evidence of efficacy, viability response, or a direct target."
        ),
        "outputs": [
            "profile_scores_official_conditions.csv",
            "condition_level_measured_scores.csv",
            "official_cell_level_measured_scores.csv",
            "candidate_cancer_measured_summary_official.csv",
            "candidate_pan_cancer_measured_summary_official.csv",
            "candidate_time_measured_summary.csv",
            "candidate_dose_measured_summary.csv",
            "candidate_condition_quality_coverage.csv",
            "candidate_official_cell_label_differences.csv",
            "predicted_measured_hiq_plot_data.csv",
            "predicted_measured_hiq_correlations.csv",
            "predicted_measured_hiq_correlations_by_seed.csv",
            "measured_reversal_official_conditions.png",
            "measured_reversal_official_conditions.pdf",
            "corrected_hdac_predicted_vs_measured_hiq.png",
            "corrected_hdac_predicted_vs_measured_hiq.pdf",
            "measured_lincs_figure_manifest.json",
            "audit.log",
        ],
    }
    (args.output_dir / "measured_lincs_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    primary_pan = pan_cancer_summary.loc[
        (pan_cancer_summary["quality_set"] == "hiq")
        & pan_cancer_summary["metric"].isin(PRIMARY_METRICS)
        & pan_cancer_summary["candidate"].isin(STRICT_CANDIDATES),
        [
            "candidate",
            "metric",
            "signatures",
            "official_cell_lines",
            "median_across_cancers",
            "positive_cancer_fraction",
        ],
    ]
    primary_correlations = correlations.loc[
        correlations["metric"].isin(PRIMARY_METRICS)
    ]
    report = "\n".join(
        [
            "===== OFFICIAL CONDITION / QUALITY COVERAGE =====",
            coverage.to_string(index=False),
            "",
            "===== CANDIDATE OFFICIAL CELL-LABEL DIFFERENCES =====",
            (
                cell_difference_summary.to_string(index=False)
                if len(cell_difference_summary)
                else "No candidate cell-label differences."
            ),
            "",
            "===== HiQ PRIMARY MEASURED REVERSAL =====",
            primary_pan.round(6).to_string(index=False),
            "",
            "===== CORRECTED-HDAC ENSEMBLE PREDICTED / HiQ MEASURED =====",
            primary_correlations.round(6).to_string(index=False),
            "",
            "===== MANIFEST =====",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "",
            "MEASURED LINCS FIGURES: PASSED",
        ]
    )
    (args.output_dir / "audit.log").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
