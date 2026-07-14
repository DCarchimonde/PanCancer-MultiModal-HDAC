from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem

CANDIDATES = {
    "TC-H-106": {"tc-h-106", "tch106", "tc h 106"},
    "RG2833": {"rg2833", "rg-2833", "rg 2833"},
    "Tianeptinaline": {"tianeptinaline"},
}
HDAC_PATTERN = re.compile(r"\bhdac\b|histone\s+deacetylase", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LINCS composition, split overlap, candidate membership, and HDAC annotations."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        required=True,
        help="Directory containing siginfo_beta.txt.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of this repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/revision/audit.",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


@lru_cache(maxsize=100_000)
def canonicalize_text(text: str) -> str | None:
    text = text.strip()
    if not text or text.lower() == "restricted":
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonicalize_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None
    return canonicalize_text(str(value))


def read_repurposing_table(path: Path) -> pd.DataFrame:
    header_row = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if line.startswith("pert_iname\t"):
                header_row = index
                break
    if header_row is None:
        raise ValueError(f"Could not locate pert_iname header in {path}")
    return pd.read_csv(path, sep="\t", skiprows=header_row, low_memory=False)


def derive_cell_id(frame: pd.DataFrame) -> pd.Series:
    if "cell_id" in frame.columns:
        return frame["cell_id"].astype(str)
    return frame["sig_id"].astype(str).str.split("_").str[1].fillna("UNKNOWN")


def overlap_count(left: pd.Series, right: pd.Series) -> int:
    return len(set(left.dropna().astype(str)) & set(right.dropna().astype(str)))


def scan_siginfo(path: Path, wanted_ids: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=1)
    preferred_columns = [
        "sig_id", "pert_id", "cmap_name", "pert_type", "cell_iname",
        "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
        "nsample", "distil_ids", "tas", "is_hiq", "qc_pass",
    ]
    usecols = [column for column in preferred_columns if column in header.columns]
    if "sig_id" not in usecols:
        raise ValueError("siginfo_beta.txt does not contain sig_id")

    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        chunksize=100_000,
        low_memory=False,
    ):
        selected = chunk[chunk["sig_id"].astype(str).isin(wanted_ids)]
        if not selected.empty:
            matches.append(selected.copy())

    if not matches:
        raise ValueError("No train/test signatures matched siginfo_beta.txt")
    return pd.concat(matches, ignore_index=True)


def join_unique(series: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in series.dropna() if str(value).strip()}))


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    raw = args.raw_data_dir.resolve()
    output = (args.output_dir or repo / "results" / "revision" / "audit").resolve()
    output.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(repo / "data" / "splits" / "train_dataset_drugcell.csv")
    test = pd.read_csv(repo / "data" / "splits" / "test_dataset_drugcell.csv")
    compound = pd.read_csv(
        repo / "data" / "metadata" / "compoundinfo_beta.txt",
        sep="\t",
        low_memory=False,
    )
    hub = read_repurposing_table(
        repo / "data" / "metadata" / "repurposing_drugs_20200324.txt"
    )
    ranking = pd.read_csv(repo / "results" / "revision" / "all_drug_names_check.csv")

    train_cell = derive_cell_id(train)
    test_cell = derive_cell_id(test)
    combined = pd.concat([train, test], ignore_index=True)

    dataset_summary = pd.DataFrame([
        {
            "dataset": "train",
            "profiles": len(train),
            "unique_drug_ids": train["drug_id"].nunique(),
            "unique_smiles": train["smiles"].nunique(),
            "unique_cell_lines": train_cell.nunique(),
        },
        {
            "dataset": "test",
            "profiles": len(test),
            "unique_drug_ids": test["drug_id"].nunique(),
            "unique_smiles": test["smiles"].nunique(),
            "unique_cell_lines": test_cell.nunique(),
        },
        {
            "dataset": "combined",
            "profiles": len(combined),
            "unique_drug_ids": combined["drug_id"].nunique(),
            "unique_smiles": combined["smiles"].nunique(),
            "unique_cell_lines": derive_cell_id(combined).nunique(),
        },
    ])
    dataset_summary.to_csv(output / "dataset_summary.csv", index=False)

    overlap = pd.DataFrame([
        {
            "identifier": "sig_id",
            "train_unique": train["sig_id"].nunique(),
            "test_unique": test["sig_id"].nunique(),
            "overlap_unique": overlap_count(train["sig_id"], test["sig_id"]),
        },
        {
            "identifier": "drug_id",
            "train_unique": train["drug_id"].nunique(),
            "test_unique": test["drug_id"].nunique(),
            "overlap_unique": overlap_count(train["drug_id"], test["drug_id"]),
        },
        {
            "identifier": "smiles",
            "train_unique": train["smiles"].nunique(),
            "test_unique": test["smiles"].nunique(),
            "overlap_unique": overlap_count(train["smiles"], test["smiles"]),
        },
        {
            "identifier": "cell_id",
            "train_unique": train_cell.nunique(),
            "test_unique": test_cell.nunique(),
            "overlap_unique": overlap_count(train_cell, test_cell),
        },
        {
            "identifier": "smiles_cell_pair",
            "train_unique": (train["smiles"].astype(str) + "||" + train_cell).nunique(),
            "test_unique": (test["smiles"].astype(str) + "||" + test_cell).nunique(),
            "overlap_unique": overlap_count(
                train["smiles"].astype(str) + "||" + train_cell,
                test["smiles"].astype(str) + "||" + test_cell,
            ),
        },
    ])
    overlap.to_csv(output / "split_overlap_summary.csv", index=False)

    wanted_ids = set(combined["sig_id"].astype(str))
    siginfo = scan_siginfo(raw / "siginfo_beta.txt", wanted_ids)
    membership = pd.concat([
        train[["sig_id"]].assign(split="train"),
        test[["sig_id"]].assign(split="test"),
    ])
    siginfo = siginfo.merge(membership, on="sig_id", how="left", validate="one_to_one")
    siginfo.to_csv(output / "matched_siginfo.csv", index=False)

    condition_rows = []
    for split_name, part in [("combined", siginfo), *list(siginfo.groupby("split"))]:
        row = {"split": split_name, "profiles": len(part)}
        for column in (
            "pert_id", "cmap_name", "cell_iname", "pert_dose", "pert_dose_unit",
            "pert_time", "pert_time_unit", "pert_type", "nsample", "distil_ids",
        ):
            if column in part.columns:
                row[f"unique_{column}"] = part[column].nunique(dropna=True)
        if "nsample" in part.columns:
            nsample = pd.to_numeric(part["nsample"], errors="coerce")
            row["nsample_median"] = nsample.median()
            row["nsample_min"] = nsample.min()
            row["nsample_max"] = nsample.max()
        condition_rows.append(row)
    pd.DataFrame(condition_rows).to_csv(output / "condition_summary.csv", index=False)

    ranking["canonical"] = ranking["SMILES"].map(canonicalize_smiles)
    ranking["normalized_name"] = ranking["Drug_Name"].map(normalize_name)
    compound["canonical"] = compound["canonical_smiles"].map(canonicalize_smiles)
    compound["normalized_name"] = compound["cmap_name"].map(normalize_name)
    hub["normalized_name"] = hub["pert_iname"].map(normalize_name)
    train_canonical = train["smiles"].map(canonicalize_smiles)
    test_canonical = test["smiles"].map(canonicalize_smiles)

    candidate_rows = []
    for candidate, aliases in CANDIDATES.items():
        normalized_aliases = {normalize_name(value) for value in aliases | {candidate}}
        rank_hits = ranking[ranking["normalized_name"].isin(normalized_aliases)]
        candidate_smiles = set(rank_hits["canonical"].dropna())
        compound_hits = compound[
            compound["normalized_name"].isin(normalized_aliases)
            | compound["canonical"].isin(candidate_smiles)
        ]
        candidate_smiles.update(compound_hits["canonical"].dropna())
        candidate_ids = set(compound_hits["pert_id"].dropna().astype(str))
        hub_hits = hub[hub["normalized_name"].isin(normalized_aliases)]

        train_mask = train["drug_id"].astype(str).isin(candidate_ids) | train_canonical.isin(candidate_smiles)
        test_mask = test["drug_id"].astype(str).isin(candidate_ids) | test_canonical.isin(candidate_smiles)

        candidate_rows.append({
            "candidate_label": candidate,
            "ranking_smiles": join_unique(rank_hits["SMILES"]),
            "compoundinfo_names": join_unique(compound_hits["cmap_name"]),
            "compoundinfo_pert_ids": join_unique(compound_hits["pert_id"]),
            "repurposing_hub_names": join_unique(hub_hits["pert_iname"]),
            "repurposing_hub_moa": join_unique(hub_hits["moa"]) if "moa" in hub_hits else "",
            "repurposing_hub_targets": join_unique(hub_hits["target"]) if "target" in hub_hits else "",
            "train_profiles": int(train_mask.sum()),
            "test_profiles": int(test_mask.sum()),
            "present_in_training": bool(train_mask.any()),
            "present_in_test": bool(test_mask.any()),
            "annotation_status": (
                "matched to compound metadata"
                if not compound_hits.empty or not hub_hits.empty
                else "manual identity verification required"
            ),
        })
    candidate_table = pd.DataFrame(candidate_rows)
    candidate_table.to_csv(output / "candidate_membership_summary.csv", index=False)

    hub_text = hub[[column for column in ("pert_iname", "moa", "target") if column in hub]].astype(str).agg(" | ".join, axis=1)
    hdac_hub = hub[hub_text.str.contains(HDAC_PATTERN, na=False)].copy()
    hdac_hub = hdac_hub.merge(
        compound[["pert_id", "cmap_name", "normalized_name", "canonical_smiles", "canonical"]],
        on="normalized_name",
        how="left",
    )
    hdac_rows = []
    for name, part in hdac_hub.groupby("pert_iname"):
        ids = set(part["pert_id"].dropna().astype(str))
        smiles = set(part["canonical"].dropna())
        train_mask = train["drug_id"].astype(str).isin(ids) | train_canonical.isin(smiles)
        test_mask = test["drug_id"].astype(str).isin(ids) | test_canonical.isin(smiles)
        hdac_rows.append({
            "pert_iname": name,
            "moa": join_unique(part["moa"]),
            "target": join_unique(part["target"]),
            "compoundinfo_pert_ids": join_unique(part["pert_id"]),
            "canonical_smiles": join_unique(part["canonical_smiles"]),
            "train_profiles": int(train_mask.sum()),
            "test_profiles": int(test_mask.sum()),
            "present_in_training": bool(train_mask.any()),
            "present_in_test": bool(test_mask.any()),
        })
    hdac_table = pd.DataFrame(hdac_rows).sort_values(
        ["present_in_training", "pert_iname"], ascending=[False, True]
    )
    hdac_table.to_csv(output / "hdac_compound_overlap.csv", index=False)

    report = {
        "dataset_summary": dataset_summary.to_dict("records"),
        "split_overlap": overlap.to_dict("records"),
        "candidate_membership": candidate_table.to_dict("records"),
        "hdac_compounds_annotated": len(hdac_table),
        "interpretation": [
            "The original split is pair-level and permits compounds and cell lines to overlap across subsets.",
            "Direct candidate membership must be disclosed in the revised manuscript.",
            "Unmatched or differently named compounds require manual identity verification.",
        ],
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Audit complete: {output}")
    print("Review candidate_membership_summary.csv and split_overlap_summary.csv first.")


if __name__ == "__main__":
    main()
