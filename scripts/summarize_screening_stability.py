from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, rankdata, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cross-seed and cross-model screening stability."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/screening_input"),
    )
    parser.add_argument(
        "--screening-root",
        type=Path,
        default=Path("results/revision/screening"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/screening_analysis"),
    )
    parser.add_argument("--metrics", default="legacy_wtrs,signed_wtrs")
    parser.add_argument("--top-thresholds", default="10,50,100,500")
    return parser.parse_args()


def discover_runs(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for manifest_path in sorted(
        root.glob("*/*/seed_*/screening_manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_dir"] = manifest_path.parent
        manifest["run_id"] = (
            f"{manifest['split']}__{manifest['model']}__seed_{manifest['seed']}"
        )
        runs.append(manifest)
    if not runs:
        raise FileNotFoundError(f"No screening manifests found under {root}")
    return runs


def metric_filename(metric: str) -> str:
    mapping = {
        "legacy_wtrs": "legacy_wtrs_scores.npy",
        "signed_wtrs": "signed_wtrs_scores.npy",
    }
    if metric not in mapping:
        raise ValueError(f"Unknown metric: {metric}")
    return mapping[metric]


def add_consensus_columns(
    library: pd.DataFrame,
    runs: list[dict[str, object]],
    cancer_names: list[str],
    metric: str,
    output_dir: Path,
    thresholds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compound_count = len(library)
    rank_sum = np.zeros(compound_count, dtype=np.float64)
    rank_sq_sum = np.zeros(compound_count, dtype=np.float64)
    percentile_sum = np.zeros(compound_count, dtype=np.float64)
    observations = 0
    top_counts = {
        threshold: np.zeros(compound_count, dtype=np.int64)
        for threshold in thresholds
    }
    run_pan_cancer: dict[str, np.ndarray] = {}
    model_accumulators: dict[str, dict[str, object]] = {}
    cancer_accumulators = {
        cancer: np.zeros(compound_count, dtype=np.float64)
        for cancer in cancer_names
    }
    cancer_counts = {cancer: 0 for cancer in cancer_names}

    for run in runs:
        score_path = Path(run["run_dir"]) / metric_filename(metric)
        scores = np.load(score_path, mmap_mode="r")
        if scores.shape != (compound_count, len(cancer_names)):
            raise ValueError(f"Unexpected score shape in {score_path}: {scores.shape}")
        ranks = rankdata(-np.asarray(scores), method="average", axis=0)
        percentiles = (ranks - 1.0) / max(compound_count - 1, 1)
        run_mean_percentile = percentiles.mean(axis=1)
        run_pan_cancer[str(run["run_id"])] = run_mean_percentile

        rank_sum += ranks.sum(axis=1)
        rank_sq_sum += np.square(ranks).sum(axis=1)
        percentile_sum += percentiles.sum(axis=1)
        observations += len(cancer_names)
        for threshold in thresholds:
            top_counts[threshold] += (ranks <= threshold).sum(axis=1)
        for cancer_index, cancer in enumerate(cancer_names):
            cancer_accumulators[cancer] += percentiles[:, cancer_index]
            cancer_counts[cancer] += 1

        model = str(run["model"])
        if model not in model_accumulators:
            model_accumulators[model] = {
                "sum": np.zeros(compound_count, dtype=np.float64),
                "count": 0,
            }
        model_accumulators[model]["sum"] += percentiles.sum(axis=1)
        model_accumulators[model]["count"] += len(cancer_names)

    mean_rank = rank_sum / observations
    rank_variance = np.maximum(
        rank_sq_sum / observations - np.square(mean_rank),
        0.0,
    )
    consensus = library.copy()
    consensus.insert(0, "metric", metric)
    consensus["mean_rank"] = mean_rank
    consensus["rank_sd"] = np.sqrt(rank_variance)
    consensus["mean_rank_percentile"] = percentile_sum / observations
    consensus["consensus_rank"] = rankdata(
        consensus["mean_rank_percentile"],
        method="average",
    )
    for threshold in thresholds:
        consensus[f"top{threshold}_count"] = top_counts[threshold]
        consensus[f"top{threshold}_fraction"] = top_counts[threshold] / observations
    for model, accumulator in model_accumulators.items():
        safe_name = model.replace("-", "_")
        consensus[f"mean_percentile_{safe_name}"] = (
            accumulator["sum"] / accumulator["count"]
        )
    consensus = consensus.sort_values(
        ["mean_rank_percentile", "rank_sd"],
        ascending=[True, True],
    ).reset_index(drop=True)
    consensus.to_csv(
        output_dir / f"compound_consensus_{metric}.csv",
        index=False,
    )

    cancer_rows: list[pd.DataFrame] = []
    base_columns = [
        "compound_index",
        "canonical_smiles",
        "display_name",
        "cache_name",
        "compoundinfo_name",
        "target",
        "moa",
        "name_conflict",
        "is_hdac_annotated",
    ]
    available = [column for column in base_columns if column in library.columns]
    for cancer in cancer_names:
        mean_percentile = cancer_accumulators[cancer] / cancer_counts[cancer]
        order = np.argsort(mean_percentile, kind="stable")[:500]
        frame = library.iloc[order][available].copy()
        frame.insert(0, "consensus_rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "mean_rank_percentile", mean_percentile[order])
        frame.insert(0, "cancer", cancer)
        frame.insert(0, "metric", metric)
        cancer_rows.append(frame)
    cancer_top = pd.concat(cancer_rows, ignore_index=True)
    cancer_top.to_csv(
        output_dir / f"cancer_consensus_top500_{metric}.csv",
        index=False,
    )

    agreement_rows: list[dict[str, object]] = []
    for left_id, right_id in itertools.combinations(sorted(run_pan_cancer), 2):
        rho, p_value = spearmanr(
            run_pan_cancer[left_id],
            run_pan_cancer[right_id],
        )
        agreement_rows.append({
            "metric": metric,
            "run_left": left_id,
            "run_right": right_id,
            "spearman_rho": rho,
            "p_value": p_value,
        })
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(
        output_dir / f"run_agreement_{metric}.csv",
        index=False,
    )

    population = len(consensus)
    hdac_mask = consensus["is_hdac_annotated"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    hdac_total = int(hdac_mask.sum())
    enrichment_rows = []
    for threshold in thresholds:
        selected = consensus.head(min(threshold, population))
        selected_hdac = int(
            selected["is_hdac_annotated"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
            .sum()
        )
        selected_count = len(selected)
        expected = (
            selected_count * hdac_total / population
            if population
            else np.nan
        )
        fold = (
            selected_hdac / expected
            if expected and expected > 0
            else np.nan
        )
        p_value = hypergeom.sf(
            selected_hdac - 1,
            population,
            hdac_total,
            selected_count,
        )
        enrichment_rows.append({
            "metric": metric,
            "top_k": selected_count,
            "library_compounds": population,
            "library_hdac": hdac_total,
            "observed_hdac": selected_hdac,
            "expected_hdac": expected,
            "fold_enrichment": fold,
            "hypergeometric_p": p_value,
        })
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment.to_csv(
        output_dir / f"hdac_enrichment_{metric}.csv",
        index=False,
    )

    return consensus, cancer_top, agreement, enrichment


def candidate_subset(consensus_frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(consensus_frames, ignore_index=True)
    text = (
        combined["display_name"].fillna("").astype(str)
        + " "
        + combined["cache_name"].fillna("").astype(str)
        + " "
        + combined["compoundinfo_name"].fillna("").astype(str)
        + " "
        + combined["compound_aliases"].fillna("").astype(str)
    ).str.lower()
    patterns = {
        "TC-H-106": r"tc[- ]?h[- ]?106",
        "RG2833": r"rg[- ]?2833",
        "Tianeptinaline_or_BG-1010": r"tianeptinaline|bg[- ]?1010",
        "Entinostat": r"entinostat|ms[- ]?275",
        "Vorinostat": r"vorinostat|saha",
    }
    frames = []
    for label, pattern in patterns.items():
        subset = combined[text.str.contains(pattern, regex=True, na=False)].copy()
        subset.insert(1, "candidate_label", label)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["candidate_label", "metric", "mean_rank_percentile"]
    )


def main() -> None:
    args = parse_args()
    metrics = [
        item.strip() for item in args.metrics.split(",") if item.strip()
    ]
    thresholds = [
        int(item)
        for item in args.top_thresholds.split(",")
        if item.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    library = pd.read_csv(
        args.input_dir / "screening_library.csv",
        low_memory=False,
    )
    disease_manifest = pd.read_csv(args.input_dir / "disease_manifest.csv")
    cancer_names = disease_manifest["cancer"].astype(str).tolist()
    runs = discover_runs(args.screening_root)

    all_consensus = []
    overview_rows = []
    for metric in metrics:
        consensus, _, agreement, enrichment = add_consensus_columns(
            library,
            runs,
            cancer_names,
            metric,
            args.output_dir,
            thresholds,
        )
        all_consensus.append(consensus)
        top100 = min(100, len(library))
        overview_rows.append({
            "metric": metric,
            "runs": len(runs),
            "cancers": len(cancer_names),
            "compounds": len(library),
            "median_pairwise_run_spearman": float(
                agreement["spearman_rho"].median()
            ),
            "min_pairwise_run_spearman": float(
                agreement["spearman_rho"].min()
            ),
            "top100_hdac_fold_enrichment": float(
                enrichment.loc[
                    enrichment["top_k"] == top100,
                    "fold_enrichment",
                ].iloc[0]
            ),
        })

    candidates = candidate_subset(all_consensus)
    candidates.to_csv(
        args.output_dir / "candidate_stability_report.csv",
        index=False,
    )
    overview = pd.DataFrame(overview_rows)
    overview.to_csv(
        args.output_dir / "screening_stability_overview.csv",
        index=False,
    )

    print("===== SCREENING STABILITY OVERVIEW =====")
    print(overview.round(6).to_string(index=False))
    print("\n===== NAMED CANDIDATES =====")
    display_columns = [
        column
        for column in [
            "candidate_label",
            "metric",
            "display_name",
            "compoundinfo_name",
            "consensus_rank",
            "mean_rank_percentile",
            "rank_sd",
            "top10_fraction",
            "top50_fraction",
            "top100_fraction",
            "name_conflict",
        ]
        if column in candidates.columns
    ]
    print(candidates[display_columns].round(6).to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
