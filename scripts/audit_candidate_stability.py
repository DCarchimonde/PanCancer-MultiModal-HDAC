from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lincs_candidate_validation import resolve_candidates  # noqa: E402


DEFAULT_SPLITS = [
    "pair_split",
    "leave_drug_out",
    "leave_cell_line_out",
    "scaffold_split",
    "hdac_class_holdout",
]
DEFAULT_MODELS = ["dual_stream", "fingerprint_mlp"]
DEFAULT_SEEDS = [1, 2, 3]
METRICS = ["legacy_wtrs", "signed_wtrs", "spearman_reversal"]
OOD_SPLITS = {
    "leave_drug_out",
    "leave_cell_line_out",
    "scaffold_split",
    "hdac_class_holdout",
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

CANDIDATE_ROLES = {
    "Mocetinostat": "core_candidate",
    "NCH-51": "secondary_candidate",
    "TC-H-106": "original_method_sensitive",
    "Belinostat": "class_support",
    "PCI-24781": "class_support",
    "Panobinostat": "class_support",
    "Entinostat": "reference_control",
    "Vorinostat": "reference_control",
    "RG2833": "prediction_only",
    "Tianeptinaline_or_BG-1010": "identity_conflict_excluded",
}

STRICT_MEASURED_CANDIDATES = {
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
}

ROLE_COLORS = {
    "core_candidate": "#b2182b",
    "secondary_candidate": "#ef8a62",
    "original_method_sensitive": "#fddbc7",
    "class_support": "#67a9cf",
    "reference_control": "#2166ac",
    "prediction_only": "#bdbdbd",
    "identity_conflict_excluded": "#636363",
}

SPLIT_LABELS = {
    "pair_split": "Pair",
    "leave_drug_out": "Drug",
    "leave_cell_line_out": "Cell",
    "scaffold_split": "Scaffold",
    "hdac_class_holdout": "HDAC",
}

METRIC_LABELS = {
    "legacy_wtrs": "Legacy",
    "signed_wtrs": "Signed",
    "spearman_reversal": "Spearman",
}


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit candidate rank stability across balanced split, model, seed, "
            "cancer, and reversal-metric configurations."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/screening_input")
    )
    parser.add_argument(
        "--weighted-root",
        type=Path,
        default=Path("results/revision/screening"),
    )
    parser.add_argument(
        "--spearman-root",
        type=Path,
        default=Path("results/revision/screening_spearman"),
    )
    parser.add_argument(
        "--measured-root",
        type=Path,
        default=Path("results/revision/measured_reversal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/candidate_stability"),
    )
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
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


def quantile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def quantile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def load_candidate_observations(
    library: pd.DataFrame,
    resolved: pd.DataFrame,
    cancer_names: list[str],
    splits: list[str],
    models: list[str],
    seeds: list[int],
    weighted_root: Path,
    spearman_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = resolved.loc[resolved["resolved"].astype(bool)].copy()
    if candidate_rows["candidate"].duplicated().any():
        raise RuntimeError("Resolved candidate labels are not unique")
    candidate_rows["compound_index"] = candidate_rows["compound_index"].astype(int)
    if (
        (candidate_rows["compound_index"] < 0)
        | (candidate_rows["compound_index"] >= len(library))
    ).any():
        raise RuntimeError("At least one candidate compound_index is out of range")

    candidate_labels = candidate_rows["candidate"].astype(str).tolist()
    candidate_indices = candidate_rows["compound_index"].to_numpy(dtype=np.int64)
    candidate_roles = [CANDIDATE_ROLES[label] for label in candidate_labels]
    compound_count = len(library)
    cancer_count = len(cancer_names)
    top_one_percent = max(1, int(np.ceil(compound_count * 0.01)))
    top_ten_percent = max(1, int(np.ceil(compound_count * 0.10)))

    metric_specs = {
        "legacy_wtrs": (weighted_root, "legacy_wtrs_scores.npy"),
        "signed_wtrs": (weighted_root, "signed_wtrs_scores.npy"),
        "spearman_reversal": (
            spearman_root,
            "spearman_reversal_scores.npy",
        ),
    }

    observation_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for split in splits:
        for metric in METRICS:
            root, filename = metric_specs[metric]
            for model in models:
                for seed in seeds:
                    run_dir = root / split / model / f"seed_{seed}"
                    score_path = run_dir / filename
                    manifest_path = run_dir / "screening_manifest.json"
                    if not score_path.exists() or not manifest_path.exists():
                        raise FileNotFoundError(
                            "Missing balanced candidate-stability input: "
                            f"{score_path}"
                        )

                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    expected_manifest = {
                        "split": split,
                        "model": model,
                        "seed": seed,
                        "screened_compounds": compound_count,
                        "cancers": cancer_names,
                    }
                    for key, expected in expected_manifest.items():
                        if manifest.get(key) != expected:
                            raise RuntimeError(
                                f"Manifest mismatch for {score_path}: {key}="
                                f"{manifest.get(key)!r}, expected {expected!r}"
                            )
                    checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
                    if not checkpoint_sha:
                        raise RuntimeError(
                            f"Missing checkpoint SHA-256 in {manifest_path}"
                        )

                    scores = np.load(score_path, mmap_mode="r")
                    expected_shape = (compound_count, cancer_count)
                    if scores.shape != expected_shape:
                        raise RuntimeError(
                            f"Unexpected score shape {scores.shape} in {score_path}; "
                            f"expected {expected_shape}"
                        )
                    if not np.isfinite(scores).all():
                        raise RuntimeError(f"Non-finite values in {score_path}")

                    score_sha = sha256_file(score_path)
                    recorded_score_sha = str(manifest.get("score_sha256", ""))
                    if recorded_score_sha and recorded_score_sha != score_sha:
                        raise RuntimeError(
                            f"Score SHA-256 mismatch for {score_path}"
                        )
                    inventory_rows.append(
                        {
                            "split": split,
                            "metric": metric,
                            "model": model,
                            "seed": seed,
                            "checkpoint_sha256": checkpoint_sha,
                            "score_sha256": score_sha,
                            "score_path": str(score_path),
                            "manifest_path": str(manifest_path),
                        }
                    )

                    score_array = np.asarray(scores, dtype=np.float64)
                    ranks = rankdata(
                        -score_array,
                        method="average",
                        axis=0,
                    )
                    rank_percentile = (
                        (ranks - 1.0) / max(compound_count - 1, 1)
                    )
                    library_percentile = 100.0 * (1.0 - rank_percentile)
                    selected_scores = score_array[candidate_indices]
                    selected_ranks = ranks[candidate_indices]
                    selected_percentiles = library_percentile[candidate_indices]

                    frame = pd.DataFrame(
                        {
                            "candidate": np.repeat(
                                candidate_labels, cancer_count
                            ),
                            "candidate_role": np.repeat(
                                candidate_roles, cancer_count
                            ),
                            "compound_index": np.repeat(
                                candidate_indices, cancer_count
                            ),
                            "cancer": np.tile(cancer_names, len(candidate_labels)),
                            "split": split,
                            "metric": metric,
                            "model": model,
                            "seed": seed,
                            "score": selected_scores.reshape(-1),
                            "rank": selected_ranks.reshape(-1),
                            "library_percentile": selected_percentiles.reshape(-1),
                        }
                    )
                    frame["top100"] = frame["rank"] <= 100
                    frame["top500"] = frame["rank"] <= 500
                    frame["top1_percent"] = frame["rank"] <= top_one_percent
                    frame["top10_percent"] = frame["rank"] <= top_ten_percent
                    observation_frames.append(frame)

    observations = pd.concat(observation_frames, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    checkpoint_counts = inventory.groupby(["split", "model", "seed"])[
        "checkpoint_sha256"
    ].nunique()
    if (checkpoint_counts != 1).any():
        mismatches = checkpoint_counts.loc[checkpoint_counts != 1].to_dict()
        raise RuntimeError(
            "Weighted and Spearman score files do not use identical checkpoints: "
            f"{mismatches}"
        )

    expected_inventory = len(splits) * len(METRICS) * len(models) * len(seeds)
    if len(inventory) != expected_inventory:
        raise RuntimeError(
            f"Run inventory has {len(inventory)} rows; expected {expected_inventory}"
        )
    expected_observations = expected_inventory * len(candidate_rows) * cancer_count
    if len(observations) != expected_observations:
        raise RuntimeError(
            f"Candidate observations have {len(observations)} rows; "
            f"expected {expected_observations}"
        )
    key_columns = [
        "candidate",
        "cancer",
        "split",
        "metric",
        "model",
        "seed",
    ]
    if observations.duplicated(key_columns).any():
        raise RuntimeError("Duplicate candidate stability observation keys")
    return observations, inventory


def summarize_runs(observations: pd.DataFrame) -> pd.DataFrame:
    return (
        observations.groupby(
            [
                "candidate",
                "candidate_role",
                "split",
                "metric",
                "model",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            cancers=("cancer", "nunique"),
            median_score=("score", "median"),
            median_rank=("rank", "median"),
            median_library_percentile=("library_percentile", "median"),
            mean_library_percentile=("library_percentile", "mean"),
            q25_library_percentile=("library_percentile", quantile_25),
            q75_library_percentile=("library_percentile", quantile_75),
            top100_cancer_fraction=("top100", "mean"),
            top500_cancer_fraction=("top500", "mean"),
            top1_percent_cancer_fraction=("top1_percent", "mean"),
            top10_percent_cancer_fraction=("top10_percent", "mean"),
        )
    )


def summarize_split_metric(
    observations: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        observations.groupby(
            ["candidate", "candidate_role", "split", "metric"],
            as_index=False,
        )
        .agg(
            cancer_observations=("cancer", "size"),
            cancers=("cancer", "nunique"),
            models=("model", "nunique"),
            seeds=("seed", "nunique"),
            median_rank=("rank", "median"),
            median_library_percentile=("library_percentile", "median"),
            mean_library_percentile=("library_percentile", "mean"),
            q25_library_percentile=("library_percentile", quantile_25),
            q75_library_percentile=("library_percentile", quantile_75),
            top100_observation_fraction=("top100", "mean"),
            top500_observation_fraction=("top500", "mean"),
            top1_percent_observation_fraction=("top1_percent", "mean"),
            top10_percent_observation_fraction=("top10_percent", "mean"),
        )
    )
    run_variation = (
        run_summary.groupby(
            ["candidate", "split", "metric"], as_index=False
        )
        .agg(
            run_configurations=("median_library_percentile", "size"),
            run_median_percentile_sd=("median_library_percentile", "std"),
            run_median_percentile_min=("median_library_percentile", "min"),
            run_median_percentile_max=("median_library_percentile", "max"),
        )
    )
    return summary.merge(
        run_variation,
        on=["candidate", "split", "metric"],
        how="left",
        validate="one_to_one",
    )


def measured_candidate_evidence(
    measured_path: Path,
    candidate_labels: list[str],
) -> pd.DataFrame:
    measured = pd.read_csv(measured_path)
    require_columns(
        measured,
        [
            "candidate",
            "metric",
            "signatures",
            "cell_lines",
            "median_across_cancers",
            "positive_cancer_fraction",
        ],
        "measured reversal summary",
    )
    rows: list[dict[str, object]] = []
    for candidate in candidate_labels:
        subset = measured.loc[measured["candidate"] == candidate]
        row: dict[str, object] = {
            "candidate": candidate,
            "measured_profiles": (
                int(subset["signatures"].max()) if len(subset) else 0
            ),
            "measured_cell_lines": (
                int(subset["cell_lines"].max()) if len(subset) else 0
            ),
        }
        for metric in METRICS:
            metric_rows = subset.loc[subset["metric"] == metric]
            row[f"measured_{metric}_median_across_cancers"] = (
                float(metric_rows["median_across_cancers"].iloc[0])
                if len(metric_rows)
                else np.nan
            )
            row[f"measured_{metric}_positive_cancer_fraction"] = (
                float(metric_rows["positive_cancer_fraction"].iloc[0])
                if len(metric_rows)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(
    observations: pd.DataFrame,
    run_summary: pd.DataFrame,
    split_metric_summary: pd.DataFrame,
    scope: str,
    selected_splits: set[str],
) -> pd.DataFrame:
    scoped_observations = observations.loc[
        observations["split"].isin(selected_splits)
    ]
    scoped_runs = run_summary.loc[run_summary["split"].isin(selected_splits)]
    scoped_split_metric = split_metric_summary.loc[
        split_metric_summary["split"].isin(selected_splits)
    ]

    rows: list[dict[str, object]] = []
    for (candidate, role), subset in scoped_runs.groupby(
        ["candidate", "candidate_role"], sort=True
    ):
        run_values = subset["median_library_percentile"].to_numpy(dtype=float)
        observation_subset = scoped_observations.loc[
            scoped_observations["candidate"] == candidate
        ]
        split_metric_subset = scoped_split_metric.loc[
            scoped_split_metric["candidate"] == candidate
        ]
        split_metric_values = split_metric_subset[
            "median_library_percentile"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "candidate_role": role,
                "scope": scope,
                "splits": int(subset["split"].nunique()),
                "metrics": int(subset["metric"].nunique()),
                "models": int(subset["model"].nunique()),
                "seeds": int(subset["seed"].nunique()),
                "run_configurations": len(subset),
                "median_run_library_percentile": float(np.median(run_values)),
                "mean_run_library_percentile": float(np.mean(run_values)),
                "run_library_percentile_q25": float(
                    np.percentile(run_values, 25)
                ),
                "run_library_percentile_q75": float(
                    np.percentile(run_values, 75)
                ),
                "run_library_percentile_iqr": float(
                    np.percentile(run_values, 75)
                    - np.percentile(run_values, 25)
                ),
                "run_library_percentile_sd": float(
                    np.std(run_values, ddof=1)
                ),
                "worst_split_metric_median_percentile": float(
                    np.min(split_metric_values)
                ),
                "best_split_metric_median_percentile": float(
                    np.max(split_metric_values)
                ),
                "best_worst_split_metric_gap": float(
                    np.max(split_metric_values) - np.min(split_metric_values)
                ),
                "top100_observation_fraction": float(
                    observation_subset["top100"].mean()
                ),
                "top500_observation_fraction": float(
                    observation_subset["top500"].mean()
                ),
                "top1_percent_observation_fraction": float(
                    observation_subset["top1_percent"].mean()
                ),
                "top10_percent_observation_fraction": float(
                    observation_subset["top10_percent"].mean()
                ),
                "eligible_for_primary_stability": candidate
                in STRICT_MEASURED_CANDIDATES,
            }
        )
    summary = pd.DataFrame(rows)
    summary["descriptive_rank_order"] = (
        summary["median_run_library_percentile"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary


def safe_spearman(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 3:
        return len(frame), np.nan
    x = frame.iloc[:, 0].to_numpy(dtype=float)
    y = frame.iloc[:, 1].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return len(frame), np.nan
    return len(frame), float(spearmanr(x, y).statistic)


def agreement_sets(candidate_values: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all_resolved": candidate_values,
        "strict_measured_8": candidate_values.loc[
            candidate_values.index.isin(STRICT_MEASURED_CANDIDATES)
        ],
    }


def agreement_tables(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (split, metric, model), subset in run_summary.groupby(
        ["split", "metric", "model"], sort=True
    ):
        pivot = subset.pivot(
            index="candidate",
            columns="seed",
            values="median_library_percentile",
        )
        for set_name, values in agreement_sets(pivot).items():
            for left, right in combinations(values.columns.tolist(), 2):
                count, correlation = safe_spearman(values[left], values[right])
                rows.append(
                    {
                        "agreement_type": "seed",
                        "analysis_set": set_name,
                        "split": split,
                        "metric": metric,
                        "model": model,
                        "seed": np.nan,
                        "level_a": str(left),
                        "level_b": str(right),
                        "candidates": count,
                        "spearman": correlation,
                    }
                )

    for (split, metric, seed), subset in run_summary.groupby(
        ["split", "metric", "seed"], sort=True
    ):
        pivot = subset.pivot(
            index="candidate",
            columns="model",
            values="median_library_percentile",
        )
        for set_name, values in agreement_sets(pivot).items():
            for left, right in combinations(values.columns.tolist(), 2):
                count, correlation = safe_spearman(values[left], values[right])
                rows.append(
                    {
                        "agreement_type": "model",
                        "analysis_set": set_name,
                        "split": split,
                        "metric": metric,
                        "model": "",
                        "seed": seed,
                        "level_a": str(left),
                        "level_b": str(right),
                        "candidates": count,
                        "spearman": correlation,
                    }
                )

    for (metric, model, seed), subset in run_summary.groupby(
        ["metric", "model", "seed"], sort=True
    ):
        pivot = subset.pivot(
            index="candidate",
            columns="split",
            values="median_library_percentile",
        )
        for set_name, values in agreement_sets(pivot).items():
            for left, right in combinations(values.columns.tolist(), 2):
                count, correlation = safe_spearman(values[left], values[right])
                rows.append(
                    {
                        "agreement_type": "split",
                        "analysis_set": set_name,
                        "split": "",
                        "metric": metric,
                        "model": model,
                        "seed": seed,
                        "level_a": str(left),
                        "level_b": str(right),
                        "candidates": count,
                        "spearman": correlation,
                    }
                )
    return pd.DataFrame(rows)


def summarize_agreement(agreement: pd.DataFrame) -> pd.DataFrame:
    return (
        agreement.groupby(
            ["agreement_type", "analysis_set"], as_index=False
        )
        .agg(
            comparisons=("spearman", "size"),
            finite_comparisons=("spearman", "count"),
            spearman_mean=("spearman", "mean"),
            spearman_median=("spearman", "median"),
            spearman_min=("spearman", "min"),
            spearman_max=("spearman", "max"),
        )
    )


def make_stability_figure(
    split_metric_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    candidate_order: list[str],
    splits: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    column_order = [
        f"{SPLIT_LABELS.get(split, split)}\n{METRIC_LABELS[metric]}"
        for split in splits
        for metric in METRICS
    ]
    heatmap_frame = split_metric_summary.copy()
    heatmap_frame["column"] = heatmap_frame.apply(
        lambda row: (
            f"{SPLIT_LABELS.get(row['split'], row['split'])}\n"
            f"{METRIC_LABELS[row['metric']]}"
        ),
        axis=1,
    )
    heatmap = heatmap_frame.pivot(
        index="candidate",
        columns="column",
        values="median_library_percentile",
    ).reindex(index=candidate_order, columns=column_order)
    if heatmap.isna().any().any():
        raise RuntimeError("Candidate stability heatmap contains missing cells")

    figure = plt.figure(figsize=(22, 14), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.35, 1.0],
        width_ratios=[2.4, 1.0],
    )
    heat_axis = figure.add_subplot(grid[0, :])
    image = heat_axis.imshow(
        heatmap.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
    )
    heat_axis.set_xticks(np.arange(len(column_order)))
    heat_axis.set_xticklabels(column_order, fontsize=9)
    heat_axis.set_yticks(np.arange(len(candidate_order)))
    heat_axis.set_yticklabels(candidate_order, fontsize=10)
    heat_axis.set_title(
        "A  Median library percentile across cancers, models, and seeds",
        loc="left",
        fontweight="bold",
    )
    for row in range(len(candidate_order)):
        for column in range(len(column_order)):
            value = float(heatmap.iloc[row, column])
            color = "white" if value < 45 or value > 80 else "black"
            heat_axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    for boundary in range(len(METRICS), len(column_order), len(METRICS)):
        heat_axis.axvline(boundary - 0.5, color="white", linewidth=2.0)
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.018, pad=0.01)
    colorbar.set_label("Library percentile (higher = stronger predicted reversal)")

    box_axis = figure.add_subplot(grid[1, 0])
    box_values = [
        run_summary.loc[
            run_summary["candidate"] == candidate,
            "median_library_percentile",
        ].to_numpy(dtype=float)
        for candidate in candidate_order
    ]
    boxes = box_axis.boxplot(
        box_values,
        vert=False,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
    )
    for patch, candidate in zip(boxes["boxes"], candidate_order):
        patch.set_facecolor(ROLE_COLORS[CANDIDATE_ROLES[candidate]])
        patch.set_alpha(0.85)
    box_axis.set_xlim(0, 100)
    box_axis.set_yticks(np.arange(1, len(candidate_order) + 1))
    box_axis.set_yticklabels(candidate_order, fontsize=9)
    box_axis.grid(axis="x", alpha=0.25)
    box_axis.invert_yaxis()
    box_axis.set_xlabel(
        "Run-level median library percentile (one value per split/model/seed/metric)"
    )
    box_axis.set_title(
        (
            "B  Distribution across "
            f"{int(run_summary.groupby('candidate').size().iloc[0])} "
            "balanced run configurations"
        ),
        loc="left",
        fontweight="bold",
    )

    bar_axis = figure.add_subplot(grid[1, 1])
    ood = (
        overall_summary.loc[overall_summary["scope"] == "ood_only"]
        .set_index("candidate")
        .reindex(candidate_order)
    )
    positions = np.arange(len(candidate_order))
    width = 0.36
    bar_axis.barh(
        positions - width / 2,
        ood["top500_observation_fraction"],
        height=width,
        color="#2166ac",
        label="Top 500",
    )
    bar_axis.barh(
        positions + width / 2,
        ood["top10_percent_observation_fraction"],
        height=width,
        color="#ef8a62",
        label="Top 10%",
    )
    bar_axis.set_yticks(positions)
    bar_axis.set_yticklabels(candidate_order, fontsize=9)
    bar_axis.invert_yaxis()
    bar_axis.set_xlim(0, 1)
    bar_axis.grid(axis="x", alpha=0.25)
    bar_axis.set_xlabel("Fraction of OOD cancer-level observations")
    bar_axis.set_title(
        "C  Recurrent high-rank frequency in OOD regimes",
        loc="left",
        fontweight="bold",
    )
    bar_axis.legend(frameon=False, loc="lower right")

    figure.suptitle(
        "Candidate prediction stability across split regimes and scoring rules",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.012,
        (
            "Ranks are descriptive model outputs, not efficacy validation. RG2833 is "
            "prediction-only; Tianeptinaline/BG-1010 is retained only as an "
            "identity-conflict sensitivity row and excluded from primary inference."
        ),
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        output_dir / "candidate_stability_figure.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "candidate_stability_figure.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    splits = parse_csv_list(args.splits)
    models = parse_csv_list(args.models)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    if len(splits) != len(set(splits)):
        raise RuntimeError("Duplicate split names were requested")
    if not set(OOD_SPLITS).issubset(splits):
        raise RuntimeError(
            "Candidate stability requires all four OOD/generalization splits: "
            f"{sorted(OOD_SPLITS)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    library_path = args.input_dir / "screening_library.csv"
    disease_manifest_path = args.input_dir / "disease_manifest.csv"
    library = pd.read_csv(library_path, low_memory=False)
    disease_manifest = pd.read_csv(disease_manifest_path)
    require_columns(
        library,
        ["compound_index", "canonical_smiles"],
        "screening library",
    )
    require_columns(disease_manifest, ["cancer"], "disease manifest")
    expected_indices = np.arange(len(library), dtype=np.int64)
    actual_indices = library["compound_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise RuntimeError("screening-library compound_index must be sequential")
    cancer_names = disease_manifest["cancer"].astype(str).tolist()

    resolved = resolve_candidates(library)
    require_columns(
        resolved,
        ["candidate", "resolved", "compound_index", "name_conflict"],
        "candidate identity map",
    )
    resolved_labels = set(
        resolved.loc[resolved["resolved"].astype(bool), "candidate"].astype(str)
    )
    expected_labels = set(CANDIDATE_ORDER)
    if resolved_labels != expected_labels:
        raise RuntimeError(
            f"Resolved candidate set changed: {sorted(resolved_labels)}"
        )
    tianeptinaline_conflict = resolved.loc[
        resolved["candidate"] == "Tianeptinaline_or_BG-1010",
        "name_conflict",
    ]
    if len(tianeptinaline_conflict) != 1 or not bool(
        tianeptinaline_conflict.iloc[0]
    ):
        raise RuntimeError(
            "Tianeptinaline/BG-1010 must remain marked as an identity conflict"
        )

    observations, inventory = load_candidate_observations(
        library,
        resolved,
        cancer_names,
        splits,
        models,
        seeds,
        args.weighted_root,
        args.spearman_root,
    )
    run_summary = summarize_runs(observations)
    split_metric_summary = summarize_split_metric(observations, run_summary)

    all_summary = summarize_overall(
        observations,
        run_summary,
        split_metric_summary,
        "all_balanced",
        set(splits),
    )
    ood_summary = summarize_overall(
        observations,
        run_summary,
        split_metric_summary,
        "ood_only",
        set(splits) & OOD_SPLITS,
    )
    overall_summary = pd.concat(
        [all_summary, ood_summary], ignore_index=True
    )
    measured_path = (
        args.measured_root / "candidate_pan_cancer_measured_summary.csv"
    )
    if not measured_path.exists():
        raise FileNotFoundError(
            f"Missing measured reversal summary: {measured_path}"
        )
    measured = measured_candidate_evidence(measured_path, CANDIDATE_ORDER)
    overall_summary = overall_summary.merge(
        measured,
        on="candidate",
        how="left",
        validate="many_to_one",
    )

    agreement = agreement_tables(run_summary)
    agreement_summary = summarize_agreement(agreement)

    observations.to_csv(
        args.output_dir / "candidate_rank_observations.csv", index=False
    )
    run_summary.to_csv(
        args.output_dir / "candidate_run_stability.csv", index=False
    )
    split_metric_summary.to_csv(
        args.output_dir / "candidate_stability_by_split_metric.csv",
        index=False,
    )
    overall_summary.to_csv(
        args.output_dir / "candidate_stability_overall.csv", index=False
    )
    agreement.to_csv(
        args.output_dir / "candidate_rank_agreement.csv", index=False
    )
    agreement_summary.to_csv(
        args.output_dir / "candidate_rank_agreement_summary.csv", index=False
    )
    inventory.to_csv(
        args.output_dir / "screening_run_inventory.csv", index=False
    )
    resolved.to_csv(
        args.output_dir / "candidate_identity_snapshot.csv", index=False
    )
    make_stability_figure(
        split_metric_summary,
        run_summary,
        overall_summary,
        CANDIDATE_ORDER,
        splits,
        args.output_dir,
        args.figure_dpi,
    )

    runtime = time.perf_counter() - started
    output_names = [
        "candidate_rank_observations.csv",
        "candidate_run_stability.csv",
        "candidate_stability_by_split_metric.csv",
        "candidate_stability_overall.csv",
        "candidate_rank_agreement.csv",
        "candidate_rank_agreement_summary.csv",
        "screening_run_inventory.csv",
        "candidate_identity_snapshot.csv",
        "candidate_stability_figure.png",
        "candidate_stability_figure.pdf",
        "candidate_stability_manifest.json",
        "audit.log",
    ]
    manifest = {
        "splits": splits,
        "ood_splits": sorted(set(splits) & OOD_SPLITS),
        "models": models,
        "seeds": seeds,
        "metrics": METRICS,
        "screened_compounds": len(library),
        "cancers": cancer_names,
        "resolved_candidates": CANDIDATE_ORDER,
        "score_file_inventory_rows": len(inventory),
        "candidate_rank_observations": len(observations),
        "run_configurations_per_candidate": int(
            run_summary.groupby("candidate").size().iloc[0]
        ),
        "runtime_seconds": runtime,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "screening_library_sha256": sha256_file(library_path),
        "disease_manifest_sha256": sha256_file(disease_manifest_path),
        "measured_summary_sha256": sha256_file(measured_path),
        "percentile_definition": (
            "100 * (1 - (descending_rank - 1) / (library_size - 1)); "
            "higher values indicate stronger predicted reversal rank."
        ),
        "aggregation_definition": (
            f"Run-level stability first takes the median across {len(cancer_names)} "
            "cancers for each candidate/split/model/seed/metric; cross-run "
            "distributions therefore do not treat cancer rows as independent "
            "replicates."
        ),
        "primary_inference_rule": (
            "RG2833 is prediction-only. Tianeptinaline/BG-1010 is an identity-conflict "
            "sensitivity row. Neither is eligible for primary stability inference."
        ),
        "evidence_note": (
            "Rank stability is descriptive prediction robustness and is not direct "
            "evidence of drug efficacy or viability response."
        ),
        "outputs": output_names,
    }
    (args.output_dir / "candidate_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    display_columns = [
        "candidate",
        "candidate_role",
        "descriptive_rank_order",
        "median_run_library_percentile",
        "run_library_percentile_iqr",
        "worst_split_metric_median_percentile",
        "top500_observation_fraction",
        "top10_percent_observation_fraction",
        "measured_profiles",
    ]
    display = overall_summary.loc[
        overall_summary["scope"] == "ood_only", display_columns
    ].sort_values("descriptive_rank_order")
    report = "\n".join(
        [
            "===== OOD CANDIDATE STABILITY =====",
            display.round(6).to_string(index=False),
            "",
            "===== SEED / MODEL / SPLIT RANK AGREEMENT =====",
            agreement_summary.round(6).to_string(index=False),
            "",
            "===== MANIFEST =====",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "",
            "CANDIDATE STABILITY AUDIT: PASSED",
        ]
    )
    (args.output_dir / "audit.log").write_text(
        report + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
