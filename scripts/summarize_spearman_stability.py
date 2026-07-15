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
        description="Summarize cross-seed and cross-model robustness of original Spearman reversal rankings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/screening_input"),
    )
    parser.add_argument(
        "--screening-root",
        type=Path,
        default=Path("results/revision/screening_spearman"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/spearman_analysis"),
    )
    parser.add_argument("--top-thresholds", default="10,50,100,500,1000")
    return parser.parse_args()


def discover_runs(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("*/*/seed_*/screening_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_dir"] = manifest_path.parent
        manifest["run_id"] = (
            f"{manifest['split']}__{manifest['model']}__seed_{manifest['seed']}"
        )
        runs.append(manifest)
    if not runs:
        raise FileNotFoundError(f"No Spearman screening manifests found under {root}")
    return runs


def main() -> None:
    args = parse_args()
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
    cancer_names = pd.read_csv(
        args.input_dir / "disease_manifest.csv"
    )["cancer"].astype(str).tolist()
    runs = discover_runs(args.screening_root)

    compound_count = len(library)
    observations = len(runs) * len(cancer_names)
    rank_sum = np.zeros(compound_count, dtype=np.float64)
    rank_sq_sum = np.zeros(compound_count, dtype=np.float64)
    percentile_sum = np.zeros(compound_count, dtype=np.float64)
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
        score_path = Path(run["run_dir"]) / "spearman_reversal_scores.npy"
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
    consensus.insert(0, "metric", "spearman_reversal")
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
        consensus[f"mean_percentile_{model}"] = (
            accumulator["sum"] / accumulator["count"]
        )
    consensus = consensus.sort_values(
        ["mean_rank_percentile", "rank_sd"],
        ascending=[True, True],
    ).reset_index(drop=True)
    consensus.to_csv(
        args.output_dir / "compound_consensus_spearman_reversal.csv",
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
        "hdac_annotation_source",
    ]
    available = [column for column in base_columns if column in library.columns]
    for cancer in cancer_names:
        mean_percentile = cancer_accumulators[cancer] / cancer_counts[cancer]
        order = np.argsort(mean_percentile, kind="stable")[:500]
        frame = library.iloc[order][available].copy()
        frame.insert(0, "consensus_rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "mean_rank_percentile", mean_percentile[order])
        frame.insert(0, "cancer", cancer)
        frame.insert(0, "metric", "spearman_reversal")
        cancer_rows.append(frame)
    pd.concat(cancer_rows, ignore_index=True).to_csv(
        args.output_dir / "cancer_consensus_top500_spearman_reversal.csv",
        index=False,
    )

    agreement_rows: list[dict[str, object]] = []
    for left_id, right_id in itertools.combinations(sorted(run_pan_cancer), 2):
        rho, p_value = spearmanr(
            run_pan_cancer[left_id],
            run_pan_cancer[right_id],
        )
        agreement_rows.append({
            "metric": "spearman_reversal",
            "run_left": left_id,
            "run_right": right_id,
            "spearman_rho": rho,
            "p_value": p_value,
        })
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(
        args.output_dir / "run_agreement_spearman_reversal.csv",
        index=False,
    )

    hdac_mask = consensus["is_hdac_annotated"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    hdac_total = int(hdac_mask.sum())
    enrichment_rows = []
    for threshold in thresholds:
        selected = consensus.head(min(threshold, compound_count))
        selected_hdac = int(
            selected["is_hdac_annotated"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
            .sum()
        )
        selected_count = len(selected)
        expected = selected_count * hdac_total / compound_count
        fold = selected_hdac / expected if expected > 0 else np.nan
        p_value = hypergeom.sf(
            selected_hdac - 1,
            compound_count,
            hdac_total,
            selected_count,
        )
        enrichment_rows.append({
            "metric": "spearman_reversal",
            "top_k": selected_count,
            "library_compounds": compound_count,
            "library_hdac": hdac_total,
            "observed_hdac": selected_hdac,
            "expected_hdac": expected,
            "fold_enrichment": fold,
            "hypergeometric_p": p_value,
        })
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment.to_csv(
        args.output_dir / "hdac_enrichment_spearman_reversal.csv",
        index=False,
    )

    text = (
        consensus["display_name"].fillna("").astype(str)
        + " "
        + consensus["cache_name"].fillna("").astype(str)
        + " "
        + consensus["compoundinfo_name"].fillna("").astype(str)
        + " "
        + consensus["compound_aliases"].fillna("").astype(str)
    ).str.lower()
    patterns = {
        "TC-H-106": r"tc[- ]?h[- ]?106",
        "RG2833": r"rg[- ]?2833",
        "Tianeptinaline_or_BG-1010": r"tianeptinaline|bg[- ]?1010",
        "Belinostat": r"belinostat",
        "PCI-24781": r"pci[- ]?24781|abexinostat",
        "Panobinostat": r"panobinostat|lbh[- ]?589",
        "Mocetinostat": r"mocetinostat|mgcd[- ]?0103",
        "Vorinostat": r"vorinostat|saha",
        "Entinostat": r"entinostat|ms[- ]?275",
    }
    candidate_frames = []
    for label, pattern in patterns.items():
        subset = consensus[text.str.contains(pattern, regex=True, na=False)].copy()
        subset.insert(1, "candidate_label", label)
        candidate_frames.append(subset)
    candidates = pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["candidate_label", "mean_rank_percentile"]
    )
    candidates.to_csv(
        args.output_dir / "candidate_stability_spearman_reversal.csv",
        index=False,
    )

    overview = pd.DataFrame([{
        "metric": "spearman_reversal",
        "runs": len(runs),
        "cancers": len(cancer_names),
        "compounds": compound_count,
        "median_pairwise_run_spearman": float(agreement["spearman_rho"].median()),
        "min_pairwise_run_spearman": float(agreement["spearman_rho"].min()),
        "top100_hdac_fold_enrichment": float(
            enrichment.loc[enrichment["top_k"] == 100, "fold_enrichment"].iloc[0]
        ),
        "top500_hdac_fold_enrichment": float(
            enrichment.loc[enrichment["top_k"] == 500, "fold_enrichment"].iloc[0]
        ),
        "top500_hdac_p": float(
            enrichment.loc[enrichment["top_k"] == 500, "hypergeometric_p"].iloc[0]
        ),
    }])
    overview.to_csv(
        args.output_dir / "spearman_stability_overview.csv",
        index=False,
    )

    print("===== SPEARMAN STABILITY OVERVIEW =====")
    print(overview.round(6).to_string(index=False))
    print("\n===== HDAC ENRICHMENT =====")
    print(enrichment.round(6).to_string(index=False))
    display_columns = [
        "candidate_label",
        "display_name",
        "compoundinfo_name",
        "consensus_rank",
        "mean_rank_percentile",
        "rank_sd",
        "top100_fraction",
        "top500_fraction",
        "name_conflict",
    ]
    print("\n===== NAMED CANDIDATES =====")
    print(candidates[display_columns].round(6).to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
