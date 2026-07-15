from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


CACHE_METADATA = Path("cache/revision/cache_metadata.csv")
SCREENING_LIBRARY = Path("data/screening_input/screening_library.csv")
SPLIT_ROOT = Path("data/splits/generalization")
BENCHMARK_ROOT = Path("results/revision/benchmarks")
OUTPUT_DIR = Path("results/revision/lincs_candidate_validation")

CANDIDATES = {
    "Mocetinostat": r"^mocetinostat$",
    "TC-H-106": r"^tc[- ]?h[- ]?106$",
    "NCH-51": r"^nch[- ]?51$",
    "Belinostat": r"^belinostat$",
    "PCI-24781": r"^pci[- ]?24781$|^abexinostat$",
    "Panobinostat": r"^panobinostat$|^lbh[- ]?589$",
    "Entinostat": r"^entinostat$|^ms[- ]?275$",
    "Vorinostat": r"^vorinostat$|^saha$",
    "RG2833": r"^rg[- ]?2833$",
    "Tianeptinaline_or_BG-1010": r"^tianeptinaline$|^bg[- ]?1010$",
}

SPLITS = [
    "pair_split",
    "leave_drug_out",
    "leave_cell_line_out",
    "scaffold_split",
    "hdac_class_holdout",
]
MODELS = ["dual_stream", "fingerprint_mlp"]


def split_values(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        part.strip()
        for part in str(value).split("|")
        if part.strip() and part.strip().lower() not in {"nan", "none"}
    }


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def resolve_candidates(library: pd.DataFrame) -> pd.DataFrame:
    fields = [
        library.get("display_name", pd.Series("", index=library.index)),
        library.get("cache_name", pd.Series("", index=library.index)),
        library.get("compoundinfo_name", pd.Series("", index=library.index)),
    ]
    normalized_fields = [field.fillna("").astype(str).str.strip().str.lower() for field in fields]

    rows: list[dict[str, object]] = []
    for label, pattern in CANDIDATES.items():
        mask = pd.Series(False, index=library.index)
        for field in normalized_fields:
            mask |= field.str.contains(pattern, regex=True, na=False)
        matches = library.loc[mask].copy()
        if matches.empty:
            rows.append({
                "candidate": label,
                "resolved": False,
                "compound_index": np.nan,
                "display_name": "",
                "compoundinfo_name": "",
                "canonical_smiles": "",
                "drug_ids": "",
                "is_hdac_annotated": False,
                "name_conflict": False,
            })
            continue

        exact_display = normalized_fields[0].loc[matches.index].str.contains(pattern, regex=True, na=False)
        chosen = matches.loc[exact_display].iloc[0] if exact_display.any() else matches.iloc[0]
        drug_ids = set()
        for column in ["source_drug_ids", "compoundinfo_pert_id"]:
            if column in chosen.index:
                drug_ids.update(split_values(chosen[column]))

        rows.append({
            "candidate": label,
            "resolved": True,
            "compound_index": int(chosen["compound_index"]),
            "display_name": str(chosen.get("display_name", "")),
            "compoundinfo_name": str(chosen.get("compoundinfo_name", "")),
            "canonical_smiles": str(chosen.get("canonical_smiles", "")),
            "drug_ids": " | ".join(sorted(drug_ids)),
            "is_hdac_annotated": bool_value(chosen.get("is_hdac_annotated", False)),
            "name_conflict": bool_value(chosen.get("name_conflict", False)),
        })
    return pd.DataFrame(rows)


def candidate_maps(resolved: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    drug_map: dict[str, str] = {}
    smiles_map: dict[str, str] = {}
    for row in resolved.itertuples(index=False):
        if not row.resolved:
            continue
        for drug_id in split_values(row.drug_ids):
            drug_map[drug_id] = row.candidate
        if row.canonical_smiles:
            smiles_map[str(row.canonical_smiles)] = row.candidate
    return drug_map, smiles_map


def assign_candidates(
    frame: pd.DataFrame,
    drug_map: dict[str, str],
    smiles_map: dict[str, str],
) -> pd.Series:
    assigned = pd.Series(pd.NA, index=frame.index, dtype="object")
    if "drug_id" in frame.columns:
        assigned = frame["drug_id"].astype(str).map(drug_map)
    if "canonical_smiles" in frame.columns:
        missing = assigned.isna()
        assigned.loc[missing] = frame.loc[missing, "canonical_smiles"].astype(str).map(smiles_map)
    return assigned


def parse_time(sig_id: str) -> str:
    match = re.search(r"_(\d+(?:\.\d+)?)H:", str(sig_id), flags=re.IGNORECASE)
    return f"{match.group(1)}H" if match else "unknown"


def parse_dose(sig_id: str) -> str:
    text = str(sig_id)
    return text.rsplit(":", 1)[-1] if ":" in text else "unknown"


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in values if pd.notna(value) and str(value)}))


def coverage_table(
    cache: pd.DataFrame,
    resolved: pd.DataFrame,
    drug_map: dict[str, str],
    smiles_map: dict[str, str],
) -> pd.DataFrame:
    frame = cache.copy()
    frame["candidate"] = assign_candidates(frame, drug_map, smiles_map)
    frame = frame.dropna(subset=["candidate"]).copy()
    frame["time"] = frame["sig_id"].map(parse_time)
    frame["dose"] = frame["sig_id"].map(parse_dose)

    rows = []
    for candidate in resolved["candidate"]:
        subset = frame.loc[frame["candidate"] == candidate]
        rows.append({
            "candidate": candidate,
            "lincs_signatures": int(len(subset)),
            "unique_drug_ids": int(subset["drug_id"].astype(str).nunique()) if len(subset) else 0,
            "cell_lines": int(subset["cell_id"].astype(str).nunique()) if len(subset) else 0,
            "cell_line_names": join_unique(subset["cell_id"]) if len(subset) else "",
            "times": join_unique(subset["time"]) if len(subset) else "",
            "doses": join_unique(subset["dose"]) if len(subset) else "",
            "tas_median": float(pd.to_numeric(subset["tas"], errors="coerce").median()) if len(subset) and "tas" in subset.columns else np.nan,
            "cache_rows_min": int(subset["cache_row"].min()) if len(subset) else np.nan,
            "cache_rows_max": int(subset["cache_row"].max()) if len(subset) else np.nan,
        })
    return resolved.merge(pd.DataFrame(rows), on="candidate", how="left")


def classify_membership(split: str, train_count: int, test_count: int) -> str:
    if test_count == 0:
        return "not_in_test"
    if split == "hdac_class_holdout" and train_count == 0:
        return "A_strict_unseen_HDAC"
    if split == "leave_drug_out" and train_count == 0:
        return "A_strict_unseen_drug"
    if split == "scaffold_split":
        return "A_unseen_scaffold"
    if split == "leave_cell_line_out":
        return "B_unseen_cell_line"
    if split == "pair_split":
        return "B_heldout_drug_cell_pair"
    return "test_present"


def split_membership_table(
    resolved: pd.DataFrame,
    drug_map: dict[str, str],
    smiles_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        train_path = SPLIT_ROOT / split / "train.csv"
        test_path = SPLIT_ROOT / split / "test.csv"
        if not train_path.exists() or not test_path.exists():
            continue
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        train["candidate"] = assign_candidates(train, drug_map, smiles_map)
        test["candidate"] = assign_candidates(test, drug_map, smiles_map)

        for candidate in resolved["candidate"]:
            train_subset = train.loc[train["candidate"] == candidate]
            test_subset = test.loc[test["candidate"] == candidate]
            train_count = int(len(train_subset))
            test_count = int(len(test_subset))
            train_drugs = set(train_subset.get("drug_id", pd.Series(dtype=str)).astype(str))
            test_drugs = set(test_subset.get("drug_id", pd.Series(dtype=str)).astype(str))
            rows.append({
                "candidate": candidate,
                "split": split,
                "train_signatures": train_count,
                "test_signatures": test_count,
                "train_cell_lines": int(train_subset["cell_id"].astype(str).nunique()) if train_count else 0,
                "test_cell_lines": int(test_subset["cell_id"].astype(str).nunique()) if test_count else 0,
                "exact_drug_id_overlap": int(len(train_drugs & test_drugs)),
                "validation_level": classify_membership(split, train_count, test_count),
            })
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    if len(clean) > 1:
        half = float(t.ppf(0.975, df=len(clean) - 1) * sd / np.sqrt(len(clean)))
        return mean, sd, mean - half, mean + half
    return mean, sd, np.nan, np.nan


def holdout_profile_tables(
    drug_map: dict[str, str],
    smiles_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_path = SPLIT_ROOT / "hdac_class_holdout" / "test.csv"
    if not test_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    test = pd.read_csv(test_path, low_memory=False)
    test["candidate"] = assign_candidates(test, drug_map, smiles_map)
    candidate_test = test.dropna(subset=["candidate"]).copy()

    seed_rows: list[pd.DataFrame] = []
    for model in MODELS:
        for seed in [1, 2, 3]:
            metrics_path = (
                BENCHMARK_ROOT
                / "hdac_class_holdout"
                / model
                / f"seed_{seed}"
                / "test_profile_metrics.csv"
            )
            if not metrics_path.exists():
                continue
            metrics = pd.read_csv(metrics_path, low_memory=False)
            merged = candidate_test[["sig_id", "candidate", "drug_id", "cell_id"]].merge(
                metrics,
                on="sig_id",
                how="inner",
                validate="one_to_one",
            )
            if merged.empty:
                continue
            grouped = merged.groupby("candidate", as_index=False).agg(
                signatures=("sig_id", "nunique"),
                cell_lines=("cell_id", "nunique"),
                pearson=("pearson", "mean"),
                spearman=("spearman", "mean"),
                profile_mse=("profile_mse", "mean"),
                profile_mae=("profile_mae", "mean"),
            )
            grouped.insert(1, "model", model)
            grouped.insert(2, "seed", seed)
            seed_rows.append(grouped)

    if not seed_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_seed = pd.concat(seed_rows, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for (candidate, model), subset in per_seed.groupby(["candidate", "model"]):
        row: dict[str, object] = {
            "candidate": candidate,
            "model": model,
            "seeds": int(subset["seed"].nunique()),
            "signatures": int(subset["signatures"].max()),
            "cell_lines": int(subset["cell_lines"].max()),
        }
        for metric in ["pearson", "spearman", "profile_mse", "profile_mae"]:
            mean, sd, low, high = mean_ci(subset[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_sd"] = sd
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    library = pd.read_csv(SCREENING_LIBRARY, low_memory=False)
    cache = pd.read_csv(CACHE_METADATA, low_memory=False)
    resolved = resolve_candidates(library)
    drug_map, smiles_map = candidate_maps(resolved)

    coverage = coverage_table(cache, resolved, drug_map, smiles_map)
    membership = split_membership_table(resolved, drug_map, smiles_map)
    per_seed, profile_summary = holdout_profile_tables(drug_map, smiles_map)

    resolved.to_csv(OUTPUT_DIR / "candidate_identity_map.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "candidate_lincs_coverage.csv", index=False)
    membership.to_csv(OUTPUT_DIR / "candidate_split_membership.csv", index=False)
    per_seed.to_csv(OUTPUT_DIR / "hdac_holdout_candidate_metrics_by_seed.csv", index=False)
    profile_summary.to_csv(OUTPUT_DIR / "hdac_holdout_candidate_profile_summary.csv", index=False)

    manifest = {
        "cache_rows": int(len(cache)),
        "resolved_candidates": int(resolved["resolved"].sum()),
        "candidates_with_lincs_signatures": int((coverage["lincs_signatures"] > 0).sum()),
        "strict_hdac_holdout_candidates": int(
            (
                (membership["split"] == "hdac_class_holdout")
                & (membership["validation_level"] == "A_strict_unseen_HDAC")
            ).sum()
        ),
        "profile_models": sorted(profile_summary["model"].unique().tolist()) if not profile_summary.empty else [],
        "outputs": [
            "candidate_identity_map.csv",
            "candidate_lincs_coverage.csv",
            "candidate_split_membership.csv",
            "hdac_holdout_candidate_metrics_by_seed.csv",
            "hdac_holdout_candidate_profile_summary.csv",
        ],
    }
    (OUTPUT_DIR / "candidate_lincs_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("===== CANDIDATE LINCS COVERAGE =====")
    coverage_columns = [
        "candidate",
        "lincs_signatures",
        "cell_lines",
        "times",
        "doses",
        "is_hdac_annotated",
        "name_conflict",
    ]
    print(coverage[coverage_columns].to_string(index=False))

    print("\n===== STRICT / HELD-OUT MEMBERSHIP =====")
    compact = membership.loc[membership["test_signatures"] > 0, [
        "candidate",
        "split",
        "train_signatures",
        "test_signatures",
        "test_cell_lines",
        "validation_level",
    ]]
    print(compact.to_string(index=False))

    print("\n===== HDAC-HOLDOUT PROFILE VALIDATION =====")
    if profile_summary.empty:
        print("No candidate profile metrics were available.")
    else:
        display = profile_summary[[
            "candidate",
            "model",
            "seeds",
            "signatures",
            "cell_lines",
            "pearson_mean",
            "pearson_seed_sd",
            "spearman_mean",
            "spearman_seed_sd",
            "profile_mse_mean",
        ]].sort_values(["candidate", "model"])
        print(display.round(6).to_string(index=False))

    print("\n===== AUDIT MANIFEST =====")
    print(json.dumps(manifest, indent=2))
    print(f"\nOutputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
