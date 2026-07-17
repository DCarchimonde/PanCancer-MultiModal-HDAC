from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lincs_candidate_validation import (  # noqa: E402
    assign_candidates,
    candidate_maps,
    resolve_candidates,
)
from scripts.screen_spearman_sensitivity import (  # noqa: E402
    prepare_disease_rank_groups,
    spearman_reversal_scores,
)


EXPECTED_CANDIDATE_COUNTS = {
    "Mocetinostat": 14,
    "TC-H-106": 6,
    "NCH-51": 17,
    "Belinostat": 24,
    "PCI-24781": 19,
    "Panobinostat": 126,
    "Entinostat": 101,
    "Vorinostat": 899,
    "RG2833": 0,
    "Tianeptinaline_or_BG-1010": 6,
}

STRICT_MEASURED_CANDIDATES = {
    "Mocetinostat",
    "TC-H-106",
    "NCH-51",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
}

MODELS = ["dual_stream", "fingerprint_mlp"]
SEEDS = [1, 2, 3]

PREDICTED_METRICS = {
    "legacy_wtrs": (
        Path("results/revision/screening"),
        "legacy_wtrs_scores.npy",
    ),
    "signed_wtrs": (
        Path("results/revision/screening"),
        "signed_wtrs_scores.npy",
    ),
    "spearman_reversal": (
        Path("results/revision/screening_spearman"),
        "spearman_reversal_scores.npy",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate candidate reversal directly from measured LINCS profiles "
            "and compare it with corrected HDAC-holdout predictions."
        )
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/revision")
    )
    parser.add_argument(
        "--screening-input",
        type=Path,
        default=Path("data/screening_input"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/measured_reversal"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--spearman-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=20260717)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution when CUDA is unavailable.",
    )
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


def load_gene_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str)
    if frame.empty:
        raise RuntimeError(f"Empty gene-ID table: {path}")
    column = "gene_id" if "gene_id" in frame.columns else frame.columns[0]
    values = frame[column].astype(str).str.strip().tolist()
    if len(values) != len(set(values)):
        raise RuntimeError(f"Duplicate gene IDs in {path}")
    return values


def measured_weighted_scores(
    expression: np.ndarray,
    disease_matrix: np.ndarray,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.memmap, np.memmap, float]:
    profile_count = expression.shape[0]
    cancer_count = disease_matrix.shape[0]
    legacy_path = output_dir / "all_profile_legacy_wtrs.npy"
    signed_path = output_dir / "all_profile_signed_wtrs.npy"
    legacy_output = np.lib.format.open_memmap(
        legacy_path,
        mode="w+",
        dtype=np.float32,
        shape=(profile_count, cancer_count),
    )
    signed_output = np.lib.format.open_memmap(
        signed_path,
        mode="w+",
        dtype=np.float32,
        shape=(profile_count, cancer_count),
    )

    disease = torch.from_numpy(
        disease_matrix.astype(np.float32, copy=False)
    ).to(device)
    squared = disease.square()
    positive_weights = torch.where(disease > 0, squared, torch.zeros_like(disease))
    negative_weights = torch.where(disease < 0, squared, torch.zeros_like(disease))
    legacy_denominator = squared.sum(dim=1).clamp_min(1e-12)
    signed_denominator = disease.abs().sum(dim=1).clamp_min(1e-12)

    started = time.perf_counter()
    with torch.inference_mode():
        for start in tqdm(
            range(0, profile_count, batch_size),
            desc="measured weighted reversal",
            unit="batch",
        ):
            stop = min(start + batch_size, profile_count)
            profiles = torch.from_numpy(
                np.array(expression[start:stop], dtype=np.float32, copy=True)
            ).to(device, non_blocking=True)
            legacy = (
                torch.relu(-profiles) @ positive_weights.T
                + torch.relu(profiles) @ negative_weights.T
            ) / legacy_denominator.unsqueeze(0)
            signed = -(profiles @ disease.T) / signed_denominator.unsqueeze(0)
            legacy_output[start:stop] = legacy.cpu().numpy()
            signed_output[start:stop] = signed.cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    legacy_output.flush()
    signed_output.flush()
    return legacy_output, signed_output, runtime


def candidate_spearman_scores(
    expression: np.ndarray,
    cache_rows: np.ndarray,
    disease_matrix: np.ndarray,
    observed_mask: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    prepared = prepare_disease_rank_groups(disease_matrix, observed_mask)
    output = np.empty(
        (len(cache_rows), disease_matrix.shape[0]), dtype=np.float32
    )
    started = time.perf_counter()
    for start in tqdm(
        range(0, len(cache_rows), batch_size),
        desc="candidate measured Spearman reversal",
        unit="batch",
    ):
        stop = min(start + batch_size, len(cache_rows))
        block = np.asarray(expression[cache_rows[start:stop]], dtype=np.float32)
        output[start:stop] = spearman_reversal_scores(
            block,
            prepared,
            disease_matrix.shape[0],
        )
    return output, time.perf_counter() - started


def build_signature_long_table(
    candidate_metadata: pd.DataFrame,
    cancer_names: list[str],
    metric_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    profile_count = len(candidate_metadata)
    cancer_count = len(cancer_names)
    repeated_metadata = candidate_metadata.loc[
        candidate_metadata.index.repeat(cancer_count)
    ].reset_index(drop=True)
    repeated_metadata["cancer"] = np.tile(cancer_names, profile_count)
    frames: list[pd.DataFrame] = []
    for metric, values in metric_arrays.items():
        if values.shape != (profile_count, cancer_count):
            raise RuntimeError(f"Unexpected candidate {metric} shape: {values.shape}")
        frame = repeated_metadata.copy()
        frame["metric"] = metric
        frame["score"] = values.reshape(-1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bootstrap_median_ci(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        value = float(clean[0])
        return value, value
    indices = rng.integers(0, len(clean), size=(iterations, len(clean)))
    medians = np.median(clean[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def summarize_measured_scores(
    signature_scores: pd.DataFrame,
    bootstrap_iterations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_scores = (
        signature_scores.groupby(
            ["candidate", "cell_id", "cancer", "metric"],
            as_index=False,
        )
        .agg(
            cell_median_score=("score", "median"),
            signatures=("sig_id", "nunique"),
            tas_median=("tas", "median"),
        )
    )

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for (candidate, cancer, metric), subset in cell_scores.groupby(
        ["candidate", "cancer", "metric"], sort=True
    ):
        values = subset["cell_median_score"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_median_ci(
            values, bootstrap_iterations, rng
        )
        rows.append(
            {
                "candidate": candidate,
                "cancer": cancer,
                "metric": metric,
                "cell_lines": int(subset["cell_id"].nunique()),
                "signatures": int(subset["signatures"].sum()),
                "cell_score_mean": float(np.mean(values)),
                "cell_score_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
                "cell_score_median": float(np.median(values)),
                "cell_score_q25": float(np.percentile(values, 25)),
                "cell_score_q75": float(np.percentile(values, 75)),
                "cell_score_median_ci95_low": ci_low,
                "cell_score_median_ci95_high": ci_high,
                "positive_cell_fraction": float(np.mean(values > 0)),
            }
        )
    return cell_scores, pd.DataFrame(rows)


def measured_compound_ranks(
    cache: pd.DataFrame,
    resolved: pd.DataFrame,
    cancer_names: list[str],
    weighted_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    structures = cache["canonical_smiles"].astype(str)
    for metric, scores in weighted_arrays.items():
        for cancer_index, cancer in enumerate(cancer_names):
            frame = pd.DataFrame(
                {
                    "canonical_smiles": structures,
                    "score": np.asarray(scores[:, cancer_index], dtype=float),
                }
            )
            compound_scores = frame.groupby(
                "canonical_smiles", as_index=False
            )["score"].median()
            ranks = rankdata(-compound_scores["score"].to_numpy(), method="average")
            compound_scores["rank"] = ranks
            compound_scores["rank_percentile"] = (
                (ranks - 1.0) / max(len(compound_scores) - 1, 1)
            )
            lookup = compound_scores.set_index("canonical_smiles")
            for candidate_row in resolved.itertuples(index=False):
                smiles = str(candidate_row.canonical_smiles)
                if smiles not in lookup.index:
                    continue
                match = lookup.loc[smiles]
                rows.append(
                    {
                        "candidate": candidate_row.candidate,
                        "cancer": cancer,
                        "metric": metric,
                        "measured_compounds": len(compound_scores),
                        "measured_compound_score": float(match["score"]),
                        "measured_compound_rank": float(match["rank"]),
                        "measured_compound_rank_percentile": float(
                            match["rank_percentile"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def predicted_candidate_scores(
    library: pd.DataFrame,
    resolved: pd.DataFrame,
    cancer_names: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    run_inventory: list[dict[str, object]] = []
    candidate_indices = {
        row.candidate: int(row.compound_index)
        for row in resolved.itertuples(index=False)
        if bool(row.resolved)
    }
    compound_count = len(library)

    for metric, (root, filename) in PREDICTED_METRICS.items():
        for model in MODELS:
            for seed in SEEDS:
                run_dir = root / "hdac_class_holdout" / model / f"seed_{seed}"
                score_path = run_dir / filename
                manifest_path = run_dir / "screening_manifest.json"
                if not score_path.exists() or not manifest_path.exists():
                    raise FileNotFoundError(
                        f"Missing corrected HDAC screening output: {score_path}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                scores = np.load(score_path, mmap_mode="r")
                if scores.shape != (compound_count, len(cancer_names)):
                    raise RuntimeError(f"Unexpected score shape: {score_path} {scores.shape}")
                expected_manifest = {
                    "split": "hdac_class_holdout",
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
                run_inventory.append(
                    {
                        "metric": metric,
                        "model": model,
                        "seed": seed,
                        "checkpoint_sha256": manifest.get("checkpoint_sha256", ""),
                        "score_path": str(score_path),
                    }
                )
                for cancer_index, cancer in enumerate(cancer_names):
                    cancer_scores = np.asarray(scores[:, cancer_index], dtype=float)
                    ranks = rankdata(-cancer_scores, method="average")
                    percentiles = (ranks - 1.0) / max(compound_count - 1, 1)
                    for candidate, compound_index in candidate_indices.items():
                        rows.append(
                            {
                                "candidate": candidate,
                                "cancer": cancer,
                                "metric": metric,
                                "model": model,
                                "seed": seed,
                                "predicted_score": float(
                                    cancer_scores[compound_index]
                                ),
                                "predicted_rank": float(ranks[compound_index]),
                                "predicted_rank_percentile": float(
                                    percentiles[compound_index]
                                ),
                                "predicted_top100": bool(
                                    ranks[compound_index] <= 100
                                ),
                                "predicted_top500": bool(
                                    ranks[compound_index] <= 500
                                ),
                            }
                        )
    inventory = pd.DataFrame(run_inventory)
    checkpoint_counts = inventory.groupby(["model", "seed"])[
        "checkpoint_sha256"
    ].nunique()
    if (checkpoint_counts != 1).any():
        mismatches = checkpoint_counts.loc[checkpoint_counts != 1].to_dict()
        raise RuntimeError(
            "Weighted and Spearman screens do not use the same checkpoint: "
            f"{mismatches}"
        )
    return pd.DataFrame(rows), run_inventory


def safe_correlations(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def predicted_measured_comparison(
    predicted: pd.DataFrame,
    measured_summary: pd.DataFrame,
    measured_ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = predicted.merge(
        measured_summary[
            [
                "candidate",
                "cancer",
                "metric",
                "cell_lines",
                "signatures",
                "cell_score_median",
                "cell_score_median_ci95_low",
                "cell_score_median_ci95_high",
                "positive_cell_fraction",
            ]
        ],
        on=["candidate", "cancer", "metric"],
        how="inner",
        validate="many_to_one",
    )
    comparison = comparison.merge(
        measured_ranks[
            [
                "candidate",
                "cancer",
                "metric",
                "measured_compound_rank",
                "measured_compound_rank_percentile",
            ]
        ],
        on=["candidate", "cancer", "metric"],
        how="left",
        validate="many_to_one",
    )
    comparison["primary_strict_candidate"] = comparison["candidate"].isin(
        STRICT_MEASURED_CANDIDATES
    )

    candidate_rows: list[dict[str, object]] = []
    for (candidate, metric, model, seed), subset in comparison.groupby(
        ["candidate", "metric", "model", "seed"], sort=True
    ):
        pearson, spearman = safe_correlations(
            subset["predicted_score"].to_numpy(dtype=float),
            subset["cell_score_median"].to_numpy(dtype=float),
        )
        rank_spearman = np.nan
        rank_subset = subset.dropna(
            subset=[
                "predicted_rank_percentile",
                "measured_compound_rank_percentile",
            ]
        )
        if len(rank_subset) >= 3:
            rank_spearman = float(
                spearmanr(
                    rank_subset["predicted_rank_percentile"],
                    rank_subset["measured_compound_rank_percentile"],
                ).statistic
            )
        candidate_rows.append(
            {
                "candidate": candidate,
                "metric": metric,
                "model": model,
                "seed": seed,
                "cancers": int(subset["cancer"].nunique()),
                "predicted_measured_pearson_across_cancers": pearson,
                "predicted_measured_spearman_across_cancers": spearman,
                "predicted_measured_rank_spearman_across_cancers": rank_spearman,
            }
        )

    overall_rows: list[dict[str, object]] = []
    primary = comparison.loc[comparison["primary_strict_candidate"]].copy()
    for (metric, model, seed), subset in primary.groupby(
        ["metric", "model", "seed"], sort=True
    ):
        pearson, spearman = safe_correlations(
            subset["predicted_score"].to_numpy(dtype=float),
            subset["cell_score_median"].to_numpy(dtype=float),
        )
        rank_spearman = np.nan
        rank_subset = subset.dropna(
            subset=[
                "predicted_rank_percentile",
                "measured_compound_rank_percentile",
            ]
        )
        if len(rank_subset) >= 3:
            rank_spearman = float(
                spearmanr(
                    rank_subset["predicted_rank_percentile"],
                    rank_subset["measured_compound_rank_percentile"],
                ).statistic
            )
        overall_rows.append(
            {
                "metric": metric,
                "model": model,
                "seed": seed,
                "strict_candidates": int(subset["candidate"].nunique()),
                "candidate_cancer_pairs": len(subset),
                "predicted_measured_pearson": pearson,
                "predicted_measured_spearman": spearman,
                "predicted_measured_rank_spearman": rank_spearman,
            }
        )
    return comparison, pd.DataFrame(candidate_rows), pd.DataFrame(overall_rows)


def candidate_pan_cancer_summary(measured_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, metric), subset in measured_summary.groupby(
        ["candidate", "metric"], sort=True
    ):
        scores = subset["cell_score_median"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "metric": metric,
                "cancers": int(subset["cancer"].nunique()),
                "signatures": int(subset["signatures"].max()),
                "cell_lines": int(subset["cell_lines"].max()),
                "median_across_cancers": float(np.median(scores)),
                "mean_across_cancers": float(np.mean(scores)),
                "positive_cancer_fraction": float(np.mean(scores > 0)),
                "best_cancer": str(
                    subset.loc[subset["cell_score_median"].idxmax(), "cancer"]
                ),
                "worst_cancer": str(
                    subset.loc[subset["cell_score_median"].idxmin(), "cancer"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; rerun with --allow-cpu only if intended")

    cache_metadata_path = args.cache_dir / "cache_metadata.csv"
    expression_path = args.cache_dir / "expression_float32.npy"
    cache_gene_path = args.cache_dir / "gene_ids.csv"
    input_gene_path = args.screening_input / "gene_ids.csv"
    library_path = args.screening_input / "screening_library.csv"
    disease_path = args.screening_input / "disease_matrix.npy"
    mask_path = args.screening_input / "disease_observed_mask.npy"
    disease_manifest_path = args.screening_input / "disease_manifest.csv"

    cache = pd.read_csv(cache_metadata_path, low_memory=False)
    require_columns(
        cache,
        [
            "cache_row",
            "sig_id",
            "drug_id",
            "cell_id",
            "canonical_smiles",
        ],
        "cache metadata",
    )
    if "tas" not in cache.columns:
        cache["tas"] = np.nan
    cache["tas"] = pd.to_numeric(cache["tas"], errors="coerce")
    expected_rows = np.arange(len(cache), dtype=np.int64)
    actual_rows = cache["cache_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_rows, actual_rows):
        raise RuntimeError("cache_row must be sequential and aligned to expression")
    if cache["sig_id"].duplicated().any():
        raise RuntimeError("Duplicate sig_id values in cache metadata")

    expression = np.load(expression_path, mmap_mode="r")
    disease_matrix = np.load(disease_path).astype(np.float32, copy=False)
    observed_mask = np.load(mask_path).astype(bool, copy=False)
    disease_manifest = pd.read_csv(disease_manifest_path)
    require_columns(disease_manifest, ["cancer"], "disease manifest")
    cancer_names = disease_manifest["cancer"].astype(str).tolist()
    cache_gene_ids = load_gene_ids(cache_gene_path)
    input_gene_ids = load_gene_ids(input_gene_path)
    if cache_gene_ids != input_gene_ids:
        raise RuntimeError("Cache and disease-matrix gene IDs/order do not match")
    expected_shape = (len(cache), len(cache_gene_ids))
    if expression.shape != expected_shape:
        raise RuntimeError(
            f"Expression shape {expression.shape} != expected {expected_shape}"
        )
    disease_shape = (len(cancer_names), len(cache_gene_ids))
    if disease_matrix.shape != disease_shape or observed_mask.shape != disease_shape:
        raise RuntimeError("Disease matrix/mask shape does not match gene alignment")

    library = pd.read_csv(library_path, low_memory=False)
    require_columns(
        library,
        ["compound_index", "canonical_smiles"],
        "screening library",
    )
    if not np.array_equal(
        library["compound_index"].to_numpy(dtype=np.int64),
        np.arange(len(library), dtype=np.int64),
    ):
        raise RuntimeError("screening-library compound_index must be sequential")
    resolved = resolve_candidates(library)
    drug_map, smiles_map = candidate_maps(resolved)
    cache["candidate"] = assign_candidates(cache, drug_map, smiles_map)
    observed_counts_raw = (
        cache.dropna(subset=["candidate"])
        .groupby("candidate")["sig_id"]
        .nunique()
        .to_dict()
    )
    observed_counts = {
        candidate: int(observed_counts_raw.get(candidate, 0))
        for candidate in EXPECTED_CANDIDATE_COUNTS
    }
    if observed_counts != EXPECTED_CANDIDATE_COUNTS:
        raise RuntimeError(
            f"Candidate counts changed: expected={EXPECTED_CANDIDATE_COUNTS}, "
            f"observed={observed_counts}"
        )

    legacy, signed, gpu_runtime = measured_weighted_scores(
        expression,
        disease_matrix,
        args.output_dir,
        args.batch_size,
        device,
    )
    if not np.isfinite(legacy).all() or not np.isfinite(signed).all():
        raise RuntimeError("Measured weighted reversal produced non-finite values")

    candidate_metadata = cache.dropna(subset=["candidate"]).copy()
    candidate_metadata = candidate_metadata.sort_values("cache_row").reset_index(drop=True)
    candidate_rows = candidate_metadata["cache_row"].to_numpy(dtype=np.int64)
    spearman, spearman_runtime = candidate_spearman_scores(
        expression,
        candidate_rows,
        disease_matrix,
        observed_mask,
        args.spearman_batch_size,
    )
    np.save(args.output_dir / "candidate_spearman_reversal.npy", spearman)

    signature_columns = [
        "cache_row",
        "sig_id",
        "candidate",
        "drug_id",
        "cell_id",
        "canonical_smiles",
        "tas",
    ]
    signature_scores = build_signature_long_table(
        candidate_metadata[signature_columns],
        cancer_names,
        {
            "legacy_wtrs": np.asarray(legacy[candidate_rows]),
            "signed_wtrs": np.asarray(signed[candidate_rows]),
            "spearman_reversal": spearman,
        },
    )
    cell_scores, measured_summary = summarize_measured_scores(
        signature_scores,
        args.bootstrap_iterations,
        args.random_seed,
    )
    pan_cancer_summary = candidate_pan_cancer_summary(measured_summary)

    measured_ranks = measured_compound_ranks(
        cache,
        resolved,
        cancer_names,
        {
            "legacy_wtrs": legacy,
            "signed_wtrs": signed,
        },
    )
    predicted, run_inventory = predicted_candidate_scores(
        library, resolved, cancer_names
    )
    comparison, candidate_correlations, overall_correlations = (
        predicted_measured_comparison(
            predicted,
            measured_summary,
            measured_ranks,
        )
    )

    signature_scores.to_csv(
        args.output_dir / "candidate_signature_measured_scores.csv", index=False
    )
    cell_scores.to_csv(
        args.output_dir / "candidate_cell_measured_scores.csv", index=False
    )
    measured_summary.to_csv(
        args.output_dir / "candidate_cancer_measured_summary.csv", index=False
    )
    pan_cancer_summary.to_csv(
        args.output_dir / "candidate_pan_cancer_measured_summary.csv", index=False
    )
    measured_ranks.to_csv(
        args.output_dir / "candidate_measured_compound_ranks.csv", index=False
    )
    predicted.to_csv(
        args.output_dir / "corrected_hdac_predicted_candidate_scores.csv", index=False
    )
    comparison.to_csv(
        args.output_dir / "predicted_measured_plot_data.csv", index=False
    )
    candidate_correlations.to_csv(
        args.output_dir / "predicted_measured_candidate_correlations.csv", index=False
    )
    overall_correlations.to_csv(
        args.output_dir / "predicted_measured_overall_correlations.csv", index=False
    )
    pd.DataFrame(run_inventory).to_csv(
        args.output_dir / "prediction_run_inventory.csv", index=False
    )

    manifest = {
        "cache_profiles": len(cache),
        "candidate_profiles": len(candidate_metadata),
        "measured_candidates": int(
            pan_cancer_summary["candidate"].nunique()
        ),
        "strict_primary_candidates": sorted(STRICT_MEASURED_CANDIDATES),
        "cancers": cancer_names,
        "gene_count": len(cache_gene_ids),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "batch_size": args.batch_size,
        "spearman_batch_size": args.spearman_batch_size,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_seed": args.random_seed,
        "weighted_score_gpu_runtime_seconds": gpu_runtime,
        "candidate_spearman_runtime_seconds": spearman_runtime,
        "expression_sha256": sha256_file(expression_path),
        "cache_metadata_sha256": sha256_file(cache_metadata_path),
        "disease_matrix_sha256": sha256_file(disease_path),
        "observed_mask_sha256": sha256_file(mask_path),
        "all_profile_legacy_wtrs_sha256": sha256_file(
            args.output_dir / "all_profile_legacy_wtrs.npy"
        ),
        "all_profile_signed_wtrs_sha256": sha256_file(
            args.output_dir / "all_profile_signed_wtrs.npy"
        ),
        "candidate_spearman_reversal_sha256": sha256_file(
            args.output_dir / "candidate_spearman_reversal.npy"
        ),
        "condition_metadata_status": (
            "Official dose/time/replicate metadata is intentionally pending and "
            "will be joined by sig_id after siginfo audit."
        ),
        "aggregation_note": (
            "Candidate-cancer summaries first take medians within cell line, then "
            "summarize across cell lines to limit profile-count imbalance."
        ),
        "evidence_note": (
            "Measured reversal is experimental transcriptomic evidence, not direct "
            "evidence of efficacy or viability response."
        ),
    }
    (args.output_dir / "measured_reversal_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = "\n".join(
        [
            "===== MEASURED PAN-CANCER REVERSAL =====",
            pan_cancer_summary.round(6).to_string(index=False),
            "",
            "===== CORRECTED-HDAC PREDICTED / MEASURED COMPARISON =====",
            overall_correlations.round(6).to_string(index=False),
            "",
            "===== MANIFEST =====",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "",
            "MEASURED LINCS REVERSAL COMPUTATION: PASSED",
        ]
    )
    (args.output_dir / "audit.log").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
