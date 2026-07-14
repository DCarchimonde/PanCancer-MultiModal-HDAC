from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit


@lru_cache(maxsize=200_000)
def canonicalize_smiles(text: str) -> str | None:
    text = text.strip()
    if not text or text.lower() == "restricted":
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=200_000)
def murcko_group(canonical_smiles: str) -> str:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return f"INVALID::{canonical_smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    # Acyclic compounds have an empty Murcko scaffold. Grouping every acyclic
    # molecule together would create one artificial mega-group, so each acyclic
    # structure is treated as its own scaffold group.
    return scaffold if scaffold else f"ACYCLIC::{canonical_smiles}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create pair-level, leave-drug-out, leave-cell-line-out, scaffold, "
            "and optional HDAC-class holdout splits for the BMC revision."
        )
    )
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument(
        "--hdac-overlap-csv",
        type=Path,
        default=Path("results/revision/audit/hdac_compound_overlap.csv"),
        help="Audit output used to identify annotated HDAC compounds for class holdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/generalization"),
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def derive_cell_id(sig_id: pd.Series) -> pd.Series:
    return sig_id.astype(str).str.split("_").str[1].fillna("UNKNOWN")


def prepare_master(clean_csv: Path, metrics_file: Path) -> pd.DataFrame:
    clean = pd.read_csv(clean_csv, low_memory=False)
    required = {"sig_id", "smiles", "drug_id"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Missing required columns in clean CSV: {sorted(missing)}")

    metrics = pd.read_csv(
        metrics_file,
        sep="\t",
        usecols=["sig_id", "tas"],
        low_memory=False,
    )
    frame = clean.merge(metrics, on="sig_id", how="inner", validate="many_to_one")
    frame = frame.loc[frame["tas"] >= 0.2].copy()
    frame = frame.loc[frame["smiles"].astype(str).str.lower() != "restricted"].copy()
    frame["canonical_smiles"] = frame["smiles"].astype(str).map(canonicalize_smiles)
    frame = frame.dropna(subset=["canonical_smiles"]).copy()
    frame["cell_id"] = derive_cell_id(frame["sig_id"])
    frame["drug_cell_pair"] = frame["canonical_smiles"] + "||" + frame["cell_id"]
    frame["scaffold_group"] = frame["canonical_smiles"].map(murcko_group)
    frame = frame.drop_duplicates(subset=["sig_id"], keep="first").reset_index(drop=True)
    return frame


def group_split(
    frame: pd.DataFrame,
    group_column: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(frame, groups=frame[group_column]))
    return frame.iloc[train_index].copy(), frame.iloc[test_index].copy()


def overlap_count(train: pd.DataFrame, test: pd.DataFrame, column: str) -> int:
    return len(set(train[column].astype(str)) & set(test[column].astype(str)))


def summarize_split(
    split_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    grouping_column: str,
    seed: int | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "split": split_name,
        "grouping_column": grouping_column,
        "seed": seed,
        "train_profiles": int(len(train)),
        "test_profiles": int(len(test)),
        "train_unique_drugs": int(train["canonical_smiles"].nunique()),
        "test_unique_drugs": int(test["canonical_smiles"].nunique()),
        "train_unique_cells": int(train["cell_id"].nunique()),
        "test_unique_cells": int(test["cell_id"].nunique()),
        "train_unique_scaffolds": int(train["scaffold_group"].nunique()),
        "test_unique_scaffolds": int(test["scaffold_group"].nunique()),
        "sig_id_overlap": overlap_count(train, test, "sig_id"),
        "drug_overlap": overlap_count(train, test, "canonical_smiles"),
        "cell_overlap": overlap_count(train, test, "cell_id"),
        "pair_overlap": overlap_count(train, test, "drug_cell_pair"),
        "scaffold_overlap": overlap_count(train, test, "scaffold_group"),
    }
    return summary


def write_split(
    output_dir: Path,
    split_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "sig_id",
        "smiles",
        "canonical_smiles",
        "drug_id",
        "cell_id",
        "drug_cell_pair",
        "scaffold_group",
    ]
    train[columns].to_csv(split_dir / "train.csv", index=False)
    test[columns].to_csv(split_dir / "test.csv", index=False)
    (split_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_hdac_holdout(
    frame: pd.DataFrame,
    hdac_overlap_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not hdac_overlap_csv.exists():
        return None
    hdac = pd.read_csv(hdac_overlap_csv, low_memory=False)
    if "pert_id" not in hdac.columns:
        return None
    hdac_ids = set(hdac.loc[hdac["pert_id"].notna(), "pert_id"].astype(str))
    if not hdac_ids:
        return None
    test = frame.loc[frame["drug_id"].astype(str).isin(hdac_ids)].copy()
    train = frame.loc[~frame["drug_id"].astype(str).isin(hdac_ids)].copy()
    if train.empty or test.empty:
        return None
    return train, test


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = prepare_master(args.clean_csv, args.metrics_file)
    master.to_csv(args.output_dir / "master_filtered.csv", index=False)

    specifications = {
        "pair_split": "drug_cell_pair",
        "leave_drug_out": "canonical_smiles",
        "leave_cell_line_out": "cell_id",
        "scaffold_split": "scaffold_group",
    }

    summaries: list[dict[str, object]] = []
    for split_name, group_column in specifications.items():
        train, test = group_split(master, group_column, args.test_size, args.seed)
        summary = summarize_split(
            split_name, train, test, group_column, args.seed
        )
        write_split(args.output_dir, split_name, train, test, summary)
        summaries.append(summary)

    hdac_split = create_hdac_holdout(master, args.hdac_overlap_csv)
    if hdac_split is not None:
        train, test = hdac_split
        summary = summarize_split(
            "hdac_class_holdout", train, test, "annotated_HDAC_class", None
        )
        write_split(args.output_dir, "hdac_class_holdout", train, test, summary)
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "split_summary.csv", index=False)

    print(f"Prepared {len(master):,} quality-controlled profiles.")
    print(summary_frame.to_string(index=False))
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
