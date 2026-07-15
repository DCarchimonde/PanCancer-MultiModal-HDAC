from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


WTRS_DIR = Path("results/revision/screening_analysis")
SPEARMAN_DIR = Path("results/revision/spearman_analysis")
OUTPUT_DIR = Path("results/revision/cross_metric_analysis")


def load_metric(path: Path, metric_label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"compound_index", "mean_rank_percentile", "consensus_rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")

    keep = [
        "compound_index",
        "canonical_smiles",
        "display_name",
        "cache_name",
        "compoundinfo_name",
        "compound_aliases",
        "target",
        "moa",
        "name_conflict",
        "is_hdac_annotated",
        "hdac_annotation_source",
        "mean_rank",
        "rank_sd",
        "mean_rank_percentile",
        "consensus_rank",
        "top100_fraction",
        "top500_fraction",
        "mean_percentile_dual_stream",
        "mean_percentile_fingerprint_mlp",
    ]
    available = [column for column in keep if column in frame.columns]
    out = frame[available].copy()

    rename = {
        "mean_rank": f"mean_rank_{metric_label}",
        "rank_sd": f"rank_sd_{metric_label}",
        "mean_rank_percentile": f"percentile_{metric_label}",
        "consensus_rank": f"rank_{metric_label}",
        "top100_fraction": f"top100_fraction_{metric_label}",
        "top500_fraction": f"top500_fraction_{metric_label}",
        "mean_percentile_dual_stream": f"dual_percentile_{metric_label}",
        "mean_percentile_fingerprint_mlp": f"fp_percentile_{metric_label}",
    }
    return out.rename(columns=rename)


def coalesce_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata_roots = [
        "canonical_smiles",
        "display_name",
        "cache_name",
        "compoundinfo_name",
        "compound_aliases",
        "target",
        "moa",
        "name_conflict",
        "is_hdac_annotated",
        "hdac_annotation_source",
    ]
    for root in metadata_roots:
        candidates = [column for column in frame.columns if column == root or column.startswith(f"{root}_")]
        if not candidates:
            continue
        combined = frame[candidates[0]]
        for column in candidates[1:]:
            combined = combined.combine_first(frame[column])
        frame[root] = combined
        drop = [column for column in candidates if column != root]
        if drop:
            frame = frame.drop(columns=drop)
    return frame


def candidate_subset(frame: pd.DataFrame) -> pd.DataFrame:
    text = (
        frame["display_name"].fillna("").astype(str)
        + " "
        + frame["cache_name"].fillna("").astype(str)
        + " "
        + frame["compoundinfo_name"].fillna("").astype(str)
        + " "
        + frame["compound_aliases"].fillna("").astype(str)
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
    rows = []
    for label, pattern in patterns.items():
        subset = frame[text.str.contains(pattern, regex=True, na=False)].copy()
        if not subset.empty:
            subset.insert(1, "candidate_label", label)
            rows.append(subset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["cross_metric_rank", "candidate_label"]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    legacy = load_metric(
        WTRS_DIR / "compound_consensus_legacy_wtrs.csv",
        "legacy",
    )
    signed = load_metric(
        WTRS_DIR / "compound_consensus_signed_wtrs.csv",
        "signed",
    )
    spearman = load_metric(
        SPEARMAN_DIR / "compound_consensus_spearman_reversal.csv",
        "spearman",
    )

    merged = legacy.merge(signed, on="compound_index", how="inner", suffixes=("", "_signedmeta"))
    merged = merged.merge(spearman, on="compound_index", how="inner", suffixes=("", "_spearmanmeta"))
    merged = coalesce_metadata(merged)

    percentile_columns = [
        "percentile_legacy",
        "percentile_signed",
        "percentile_spearman",
    ]
    for column in percentile_columns:
        if column not in merged.columns:
            raise ValueError(f"Missing merged percentile column: {column}")

    values = merged[percentile_columns].to_numpy(dtype=float)
    merged["cross_metric_mean_percentile"] = values.mean(axis=1)
    merged["cross_metric_median_percentile"] = np.median(values, axis=1)
    merged["cross_metric_worst_percentile"] = values.max(axis=1)
    merged["cross_metric_best_percentile"] = values.min(axis=1)
    merged["cross_metric_range"] = values.max(axis=1) - values.min(axis=1)
    merged["cross_metric_rank"] = rankdata(
        merged["cross_metric_mean_percentile"],
        method="average",
    )

    merged = merged.sort_values(
        ["cross_metric_mean_percentile", "cross_metric_worst_percentile"],
        ascending=[True, True],
    ).reset_index(drop=True)
    merged.to_csv(OUTPUT_DIR / "cross_metric_consensus_all.csv", index=False)

    hdac_mask = merged["is_hdac_annotated"].astype(str).str.lower().isin({"true", "1"})
    hdac = merged.loc[hdac_mask].copy().sort_values("cross_metric_rank")
    hdac.to_csv(OUTPUT_DIR / "cross_metric_hdac_consensus.csv", index=False)

    candidates = candidate_subset(merged)
    candidates.to_csv(OUTPUT_DIR / "cross_metric_named_candidates.csv", index=False)

    display_columns = [
        "cross_metric_rank",
        "display_name",
        "compoundinfo_name",
        "percentile_legacy",
        "percentile_signed",
        "percentile_spearman",
        "cross_metric_mean_percentile",
        "cross_metric_worst_percentile",
        "cross_metric_range",
        "name_conflict",
    ]

    print("===== CROSS-METRIC HDAC CONSENSUS =====")
    print(hdac[display_columns].head(20).round(6).to_string(index=False))

    print("\n===== NAMED CANDIDATES ACROSS METRICS =====")
    candidate_columns = ["candidate_label", *display_columns]
    print(candidates[candidate_columns].round(6).to_string(index=False))

    print(f"\nOutputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
